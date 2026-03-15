import grpc
from concurrent import futures
import logging
import sys
import os
from grpc_reflection.v1alpha import reflection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "generated")))

from generated import ml_car_plate_recognition_pb2
from generated import ml_car_plate_recognition_pb2_grpc

# Your ML Model (replace with real model)
# class MLModel:
#     def __init__(self):
#         # Load your model here (e.g., torch.load, transformers pipeline)
#         logging.info("🧠 Loading ML model...")
#         # self.model = transformers.pipeline(...)
#         pass

#     def predict(self, text: str) -> tuple[str, float]:
#         # Replace with real inference logic
#         if "hello" in text.lower():
#             return "greeting", 0.95
#         elif "bye" in text.lower():
#             return "farewell", 0.90
#         else:
#             return "unknown", 0.30


class MLServicer(
    ml_car_plate_recognition_pb2_grpc.MLCarPlateRecognitionServiceServicer
):
    # def __init__(self):
    # self.model = model

    def RecognizeCarPlate(self, request, context):
        try:
            logging.debug("got request")

            # prediction, confidence = self.model.predict(request.input_text)

            # logging.info(f" Sending response: {prediction} ({confidence:.2f})")

            return ml_car_plate_recognition_pb2.PredictResponse(
                plate_number="AAAAAAAA", confidence=1.0
            )
        except Exception as e:
            logging.error(f" Inference error: {e}")
            return ml_car_plate_recognition_pb2.PredictResponse(
                plate_number="", confidence=0.0
            )


def serve(host: str = "0.0.0.0", port: int = 50051):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Register the servicer
    ml_car_plate_recognition_pb2_grpc.add_MLCarPlateRecognitionServiceServicer_to_server(
        MLServicer(), server
    )

    SERVICE_NAMES = (
        ml_car_plate_recognition_pb2.DESCRIPTOR.services_by_name[
            "MLCarPlateRecognitionService"
        ].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    address = f"{host}:{port}"
    server.add_insecure_port(address)

    server.start()
    logging.info(f" ML gRPC server started on {address}")
    logging.info(" Service: MLCarPlateRecognitionService")
    logging.info(" Method: RecognizeCarPlate")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        server.stop(0)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    serve()


if __name__ == "__main__":
    main()
