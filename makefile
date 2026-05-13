export GOPROXY := https://proxy.golang.org,direct
export GOSUMDB := sum.golang.org

go-deps:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

gen-go:
	mkdir -p clients/go/pb
	protoc --go_out=./clients/go/pb --go_opt=paths=source_relative --go-grpc_out=./clients/go/pb --go-grpc_opt=paths=source_relative -I proto proto/ml_car_plate_recognition.proto

gen: gen-go
	uv run python -m grpc_tools.protoc -Iproto --python_out=src/generated --grpc_python_out=src/generated proto/ml_car_plate_recognition.proto

run-app:
	uv run python src/server.py

train-ocr:
	uv run --extra training python scripts/train_ocr_v3.py

train-ocr-v4:
	uv run --extra training python scripts/train_ocr_v4.py

network:
	docker network create ml_cpr_network 2>/dev/null || true

up: network
	docker-compose up --build

down:
	docker-compose down

restart: down up

build:
	docker-compose build --no-cache

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker-network rm ml_cpr_network 2>/dev/null || true

benchmark:
	uv run --extra benchmark python scripts/benchmark_ocr.py

benchmark-d:
	uv run --extra benchmark python scripts/benchmark_detect.py
