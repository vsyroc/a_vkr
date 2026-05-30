import os
from shapely.validation import make_valid
from shapely import MultiPolygon
from shapely.geometry import Polygon, box
from PIL import Image
import numpy as np

# Функция для нормализации координат полигонов относительно нового кропа
def normalize_polygon_coords(polygon, crop_box):
    x_min, y_min, x_max, y_max = crop_box
    width, height = x_max - x_min, y_max - y_min

    normalized_coords = []
    if isinstance(polygon, Polygon):
        for x, y in polygon.exterior.coords:
            norm_x = (x - x_min) / width
            norm_y = (y - y_min) / height
            normalized_coords.append((norm_x,norm_y))
        return normalized_coords


# Функция для чтения полигонов из файла
def read_polygons(file_path):
    polygons = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = list(map(float, line.strip().split()))
            class_id = int(data[0])
            coords = [(data[i], data[i + 1]) for i in range(1, len(data), 2)]
            polygons.append((class_id, Polygon(coords)))
    return polygons


# Функция для записи полигонов в файл
def write_polygons(file_path, polygons):
    with open(file_path, "w") as file:
        for class_id, polygon in polygons:
            coords = list(polygon.exterior.coords)[:-1]  # Удаляем повторяющуюся последнюю точку
            file.write(f"{class_id} " + " ".join([f"{x} {y}" for x, y in coords]) + "\n")


# Основной код
def process_image_and_polygons(image_path, polygons_path, crop_box, output_image_path, output_polygons_path):
    # Открываем изображение и выполняем кроп
    image = Image.open(image_path)
    cropped_image = image.crop(crop_box)

    # Читаем полигоны из файла
    polygons = read_polygons(polygons_path)

    # Создаем полигон, представляющий кроп
    global_box = (0, 0, 7952, 5304)
    crop_polygon = box(*crop_box)
    crop_coordinates = normalize_polygon_coords(crop_polygon, global_box)
    crop_polygon = box(crop_coordinates[0][0], crop_coordinates[0][1], crop_coordinates[2][0], crop_coordinates[2][1])


    # Находим пересечение полигонов с кропом и нормализуем координаты
    new_polygons = []

    for class_id, polygon in polygons:

        intersection = (make_valid(polygon)).intersection(crop_polygon)
        if intersection.is_empty:
            return False
        else:
            intersections = []
            print(intersection)
            if isinstance(intersection, MultiPolygon):
                intersections = list(intersection.geoms)
            else: intersections.append(intersection)
            for intersection in intersections:
                normalized_coords = normalize_polygon_coords(intersection, ( crop_coordinates[0][0], crop_coordinates[0][1],crop_coordinates[2][0], crop_coordinates[2][1]))
                print(normalized_coords)
                normalized_coords = [(1-i[0], i[1] )for i in normalized_coords]


                new_polygons.append((class_id, Polygon(normalized_coords)))
                # Сохраняем кропнутое изображение
                cropped_image.save(output_image_path)

                # Записываем новые полигоны в файл
                write_polygons(output_polygons_path, new_polygons)
            return True


# Обработка всех изображений и файлов координат в папках
def process_all_images_and_polygons(images_folder, polygons_folder, output_images_folder,
                                    output_polygons_folder):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg', 'tif'))]
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        polygons_file = os.path.splitext(image_file)[0] + ".txt"
        polygons_path = os.path.join(polygons_folder, polygons_file)
        crop_size=1984
        i=0
        while i <20:
            x=np.random.randint(0,7952-crop_size)
            y=np.random.randint(0,5304-crop_size)
            crop_box = (x, y, x+crop_size, y+crop_size)
            if os.path.exists(polygons_path):

                output_image_path = os.path.join(output_images_folder, image_file)
                output_image_path = list(output_image_path)
                output_image_path.insert(-4, f'{i}')
                output_image_path = ''.join(output_image_path)
                output_polygons_path = os.path.join(output_polygons_folder, polygons_file)
                output_polygons_path = list(output_polygons_path)
                output_polygons_path.insert(-4, f'{i}')
                output_polygons_path = ''.join(output_polygons_path)

                if process_image_and_polygons(image_path, polygons_path, crop_box, output_image_path, output_polygons_path):i+=1

            else:
                print(f"No corresponding polygons file found for {image_file}")


# Пример использования
images_folder = r"E:\CK\pythonProject\COCO_to_YOLOv8\YOLO_dataset3\default\images"
polygons_folder = r"E:\CK\pythonProject\COCO_to_YOLOv8\YOLO_dataset3\default\labels"
output_images_folder = r"E:\CK\pythonProject\COCO_to_YOLOv8\YOLO_dataset7\default\train\images"
output_polygons_folder = r"E:\CK\pythonProject\COCO_to_YOLOv8\YOLO_dataset7\default\train\labels"

os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_polygons_folder, exist_ok=True)

process_all_images_and_polygons(images_folder, polygons_folder, output_images_folder, output_polygons_folder)