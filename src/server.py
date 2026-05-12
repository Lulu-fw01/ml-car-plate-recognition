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
from torchvision import transforms
from ultralytics import YOLO

from pl_modules.ocr_module_v3 import OCRModuleV3
from pl_modules.greedy.decode_greedy import decode_greedy

from generated import ml_car_plate_recognition_pb2
from generated import ml_car_plate_recognition_pb2_grpc


class MLServicer(
    ml_car_plate_recognition_pb2_grpc.MLCarPlateRecognitionServiceServicer
):
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        logging.info("Loading detection model...")
        self.det_model = YOLO(cfg.models.detection)
        logging.info("Loading OCR model...")

        self.model_v3 = OCRModuleV3.load_from_checkpoint(
            cfg.models.ocr, map_location=self.device, weights_only=False
        )
        self.model_v3.eval()

        self.cfg = cfg
        self.alphabet = cfg.ocr.alphabet
        self.pad_idx = cfg.ocr.pad_idx
        self.max_plates = cfg.detection.max_plates
        self.detection_threshold = self.cfg.detection.conf_threshold

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    (64, 160), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _detect_plates(
        self, image_bytes: bytes, conf_threshold: float, max_plates: int
    ):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image from bytes")

        h, w = image.shape[:2]
        logging.debug(f"Image decoded: {w}x{h}")

        results = self.det_model(image, conf=conf_threshold, imgsz=320)
        boxes = results[0].boxes

        if len(boxes) == 0:
            logging.warning(f"No detections with conf>={conf_threshold}")
            return [], image

        detections = []
        for i, box in enumerate(boxes[:max_plates]):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0].cpu().numpy())
            detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return detections, image

    def _crop_plate(self, image, coords):
        x1, y1, x2, y2 = coords
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb)

    def _recognize_text(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model_v3.model(tensor)
            log_probs = torch.log_softmax(logits, dim=-1)

        decoded = decode_greedy(self.alphabet, self.pad_idx, log_probs)
        plate_text = decoded[0] if decoded else ""

        probs = torch.softmax(logits, dim=-1).squeeze(1)
        preds = logits.argmax(dim=-1).squeeze(1)

        confidences = []
        prev = -1
        for i, idx in enumerate(preds):
            idx_item = idx.item()
            if idx_item == prev:
                continue
            if idx_item == self.pad_idx:
                prev = idx_item
                continue
            if idx_item < len(self.alphabet):
                confidences.append(probs[i, idx_item].item())
            prev = idx_item

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

    def RecognizeCarPlates(self, request, context):
        try:
            img_bytes = request.image_data
            logging.info(f"Got request: {len(img_bytes)} bytes")

            if len(img_bytes) == 0:
                logging.error("Empty image data received")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty image data")

            self._log_image_format(img_bytes)
            self._save_image(img_bytes)

            conf_threshold = self.detection_threshold

            detections, full_image = self._detect_plates(
                img_bytes, conf_threshold, self.max_plates
            )

            if len(detections) == 0:
                logging.warning("No license plates detected in the image")
                return ml_car_plate_recognition_pb2.PredictMultiResponse(
                    plates=[], total_detected=0
                )
        except grpc.RpcError:
            raise
        except Exception as e:
            logging.error(f"Detection error: {e}")
            return ml_car_plate_recognition_pb2.PredictMultiResponse(
                plates=[], total_detected=0
            )

        plates = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            try:
                pil_plate = self._crop_plate(full_image, (x1, y1, x2, y2))
                plate_number, confidence = self._recognize_text(pil_plate)
            except Exception as e:
                logging.error(f"OCR error: {e}")
                continue

            bbox = ml_car_plate_recognition_pb2.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            plates.append(
                ml_car_plate_recognition_pb2.PlateResult(
                    plate_number=plate_number, confidence=confidence, bbox=bbox
                )
            )

        return ml_car_plate_recognition_pb2.PredictMultiResponse(
            plates=plates, total_detected=len(plates)
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
    full_name = ml_car_plate_recognition_pb2.DESCRIPTOR.services_by_name[
        "MLCarPlateRecognitionService"
    ].full_name
    logging.info(f"FULL SERVICE NAME: {full_name}")
    logging.info("Method: RecognizeCarPlates")

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
