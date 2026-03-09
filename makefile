gen:
	uv run python -m grpc_tools.protoc -Iproto --python_out=src/generated --grpc_python_out=src/generated proto/ml_car_plate_recognition.proto
