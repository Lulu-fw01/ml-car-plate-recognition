# ml-car-plate-recognition

## Установка зависимостей
```bash
uv venv && source .venv/bin/activate
```

```bash
    pre-commit install
```

```bash
    pre-commit run --all-files
```

## Загрузка данных
```bash
download-data
```

## Обучение
```bash
mlflow ui --host 127.0.0.1 --port 8080
```

### Обучение ocr модели
```bash
python scripts/train_ocr.py
```

### Обучение модели детекции номеров.
```bash
python scripts/train_detect.py
```

Вызвать распознование
```bash
grpcurl -plaintext \
  -d '{"image_link": "image/link"}' \
  localhost:50051 ml.MLCarPlateRecognitionService/RecognizeCarPlate
```
