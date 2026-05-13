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
