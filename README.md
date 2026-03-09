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

python scripts/inference.py image_path /Users/lumarkov/Downloads/nom1.jpg --detect-path mlartifacts/218476399212749806/02a1439efc5b46c3a40277e99e7e30d6/artifacts/models/best.pt \
  --ocr-path mlartifacts/ocr/ocr-v1-epoch=06-val_acc=0.9588.ckpt


## Baseline

### Детекция номера на изображении

yolo5s

### Распознование номера
