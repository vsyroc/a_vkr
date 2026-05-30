import os
from PIL import Image

from app.config import PROCESSED_DIR, UPLOADS_DIR


def Tiff():
    # Список исходных папок и целевых папок
    source_folders = [UPLOADS_DIR]

    target_folders = [PROCESSED_DIR]

    # Проверяем, что количество исходных и целевых папок совпадает
    if len(source_folders) != len(target_folders):
        print("Количество исходных и целевых папок должно быть одинаковым.")
        exit()

    # Проходим по каждой паре исходная папка - целевая папка
    for source_folder, target_folder in zip(source_folders, target_folders):
        # Создаем целевую папку, если она не существует
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Получаем список файлов изображений в исходной папке
        image_files = os.listdir(source_folder)

        # Проходим по каждому файлу изображения
        for image_file in image_files:
            # Полный путь к исходному файлу
            source_path = os.path.join(source_folder, image_file)

            # Загружаем изображение
            try:
                image = Image.open(source_path)
            except Exception as e:
                print(f"Ошибка при открытии файла {source_path}: {e}")
                continue

            # Проверяем цветовое пространство изображения (должно быть RGB)
            if image.mode != "RGB":
                image = image.convert("RGB")  # Преобразуем в RGB, если необходимо

            # Создаем путь для сохранения в целевой папке с тем же именем файла, но с расширением .tif
            target_path = os.path.join(
                target_folder, os.path.splitext(image_file)[0] + ".tif"
            )

            # Сохраняем изображение в формате TIFF с глубиной цвета 24 бита (8 бит на канал)
            try:
                image.save(target_path, "TIFF")
                print(f"Изображение успешно сохранено в {target_path}")
            except Exception as e:
                print(f"Ошибка при сохранении изображения {source_path} в TIFF: {e}")

    print(
        "Преобразование изображений в формат TIFF с глубиной цвета 24 бита завершено."
    )
