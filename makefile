export GOPROXY := https://proxy.golang.org,direct
export GOSUMDB := sum.golang.org

go-deps:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

gen:
	uv run python -m grpc_tools.protoc -Iproto --python_out=src/generated --grpc_python_out=src/generated proto/ml_car_plate_recognition.proto

gen-go:
	mkdir -p clients/go/pb
	protoc --go_out=./clients/go/pb --go_opt=paths=source_relative --go-grpc_out=./clients/go/pb --go-grpc_opt=paths=source_relative -I proto proto/ml_car_plate_recognition.proto
