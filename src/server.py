import grpc
from concurrent import futures
import logging
import sys
import os
import time
from grpc_reflection.v1alpha import reflection
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "generated")))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

from pl_modules.ocr_module_v1 import OCRModuleV1

from generated import ml_car_plate_recognition_pb2
from generated import ml_car_plate_recognition_pb2_grpc


class MLServicer(
    ml_car_plate_recognition_pb2_grpc.MLCarPlateRecognitionServiceServicer
):
    def __init__(self, cfg):
        logging.info("Loading detection model...")
        self.det_model = YOLO(cfg.models.detection)
        logging.info("Loading OCR model...")

        self.ocr_model = OCRModuleV1.load_from_checkpoint(
            cfg.models.ocr, map_location="cpu", weights_only=False
        )
        self.ocr_model.eval()

        self.cfg = cfg
        self.alphabet = cfg.ocr.alphabet
        self.pad_idx = cfg.ocr.pad_idx
        self.img_h = cfg.ocr.img_h
        self.img_w = cfg.ocr.img_w

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((self.img_h, self.img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _detect_plate(self, image_bytes: bytes, conf_threshold: float = 0.5):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image from bytes")

        h, w = image.shape[:2]
        logging.debug(f"Image decoded: {w}x{h}")

        results = self.det_model(image, conf=conf_threshold)
        boxes = results[0].boxes

        if len(boxes) == 0:
            logging.warning(f"No detections with conf>={conf_threshold}")
            return None, image

        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return (x1, y1, x2, y2), image

    def _crop_plate(self, image, coords):
        x1, y1, x2, y2 = coords
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb)

    def _recognize_text(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            logits = self.ocr_model.model(tensor)
            preds = logits.argmax(dim=-1).squeeze()
            max_probs = F.softmax(logits, dim=-1).max(dim=-1).values.squeeze(0)

        chars = []
        confidences = []
        for i, idx in enumerate(preds):
            idx_item = idx.item()
            if idx_item == self.pad_idx:
                continue
            chars.append(self.alphabet[idx_item])
            confidences.append(max_probs[i].item())

        plate_text = "".join(chars)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return plate_text, avg_confidence

    def _save_image(self, img_bytes):
        os.makedirs("debug_images", exist_ok=True)
        debug_path = f"debug_images/request_{int(time.time())}.jpg"
        with open(debug_path, "wb") as f:
            f.write(img_bytes)
        logging.info(f"Saved debug image to {debug_path}")

    def _log_image_format(self, img_bytes):
        if img_bytes[:2] == b"\xff\xd8":
            fmt = "JPEG"
        elif img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            fmt = "PNG"
        elif img_bytes[:4] == b"RIFF":
            fmt = "WEBP"
        else:
            fmt = f"UNKNOWN (starts with {img_bytes[:10]})"

        logging.info(f"Image format detected: {fmt}")

    def RecognizeCarPlate(self, request, context):
        try:
            img_bytes = request.image_data
            logging.info(f"Got request: {len(img_bytes)} bytes")

            if len(img_bytes) == 0:
                logging.error("Empty image data received")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty image data")

            self._log_image_format(img_bytes)
            self._save_image(img_bytes)

            coords, full_image = self._detect_plate(
                img_bytes, conf_threshold=self.cfg.detection.conf_threshold
            )

            if coords is None:
                logging.warning("No license plate detected in the image")
                context.abort(
                    grpc.StatusCode.NOT_FOUND, "No license plate detected in the image"
                )

            pil_plate = self._crop_plate(full_image, coords)
            plate_number, confidence = self._recognize_text(pil_plate)

            logging.info(
                f"Recognized plate: {plate_number} (confidence: {confidence:.2f})"
            )

            return ml_car_plate_recognition_pb2.PredictResponse(
                plate_number=plate_number, confidence=confidence
            )
        except grpc.RpcError:
            raise
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return ml_car_plate_recognition_pb2.PredictResponse(
                plate_number="", confidence=0.0
            )


def serve(cfg):
    options = [
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)

    ml_car_plate_recognition_pb2_grpc.add_MLCarPlateRecognitionServiceServicer_to_server(
        MLServicer(cfg), server
    )

    SERVICE_NAMES = (
        ml_car_plate_recognition_pb2.DESCRIPTOR.services_by_name[
            "MLCarPlateRecognitionService"
        ].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    address = f"{cfg.server.host}:{cfg.server.port}"
    server.add_insecure_port(address)

    server.start()
    logging.info(f"ML gRPC server started on {address}")
    logging.info("Service: MLCarPlateRecognitionService")
    logging.info("Method: RecognizeCarPlate")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        server.stop(0)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    @hydra.main(version_base=None, config_path="../configs", config_name="server")
    def run(cfg: DictConfig):
        serve(cfg)

    run()


if __name__ == "__main__":
    main()
