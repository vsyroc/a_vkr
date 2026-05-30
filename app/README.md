# ВКР — Прототип системы обработки изображений

## Запуск

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Откройте http://127.0.0.1:8000

## Структура

```
vkr-app/
├── main.py              # FastAPI-приложение
├── requirements.txt
├── templates/
│   └── index.html       # Jinja2-шаблон (одностраничный интерфейс)
├── uploads/             # Оригинальные загруженные изображения
└── processed/           # Обработанные изображения (результат нейронки)
```

## Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Главная страница |
| POST | `/api/process` | Принимает `multipart/form-data` с полем `file`, сохраняет оригинал и «обработанную» копию, возвращает JSON с URL обоих файлов |

### Пример ответа `/api/process`
```json
{
  "original_url": "/uploads/abc123.jpg",
  "processed_url": "/processed/abc123_processed.jpg"
}
```

## Подключение нейронной сети

В `main.py` найдите блок:
```python
# --- ЗАГЛУШКА НЕЙРОННОЙ СЕТИ ---
# Здесь будет вызов модели. Пока просто копируем файл как «результат».
processed_path = PROCESSED_DIR / f"{file_id}_processed{ext}"
shutil.copy(original_path, processed_path)
# --------------------------------
```
Замените `shutil.copy(...)` на вызов вашей модели, которая читает `original_path` и записывает результат в `processed_path`.
