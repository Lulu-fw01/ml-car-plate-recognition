import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "generated"))

import grpc
import cv2
import argparse

from generated import ml_car_plate_recognition_pb2
from generated import ml_car_plate_recognition_pb2_grpc


def test_recognition(image_path: str, host: str = "localhost", port: int = 50051):
    with open(image_path, "rb") as f:
        image_data = f.read()

    print(f"Image loaded: {image_path} ({len(image_data)} bytes)")

    img = cv2.imread(image_path)
    if img is not None:
        print(f"Image shape: {img.shape}")

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = ml_car_plate_recognition_pb2_grpc.MLCarPlateRecognitionServiceStub(channel)

    print(f"Sending request to {host}:{port}...")

    try:
        response = stub.RecognizeCarPlate(
            ml_car_plate_recognition_pb2.PredictRequest(image_data=image_data)
        )

        print("\nResult:")
        print(f"  Plate:     {response.plate_number}")
        print(f"  Confidence: {response.confidence:.2%}")

    except grpc.RpcError as e:
        print(f"\nError: {e.code()} - {e.details()}")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="Test car plate recognition server")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=50051, help="Server port (default: 50051)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        return 1

    return test_recognition(args.image, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
