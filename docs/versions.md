# versions.md
cat > docs/versions.md << 'EOF'
# Фиксация версий

## Библиотеки
- Python: 3.11.9
- PyTorch: 2.5.1+cu124
- Ultralytics: 8.3.40
- OpenCV: 4.10.0.84
- NumPy: 1.26.4
- Albumentations: 1.4.20

## Исходные веса
- Модель: `yolov8m.pt`
- Источник: pretrained on COCO
- URL: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt

## Датасет
- GTSDB: German Traffic Sign Detection Benchmark
- URL: http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
- Дата скачивания: [указать]
- Размер: 900 изображений, 1213 аннотаций

## Hardware
- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
EOF

# annotation_protocol.md
cat > docs/annotation_protocol.md << 'EOF'
# Протокол подготовки данных

## Маппинг классов GTSDB

Исходные 43 класса GTSDB агрегированы в 4 укрупнённые категории:

### Prohibitory (запрещающие) - 557 объектов
- 0-5, 7-10, 15, 16: speed limits, no overtaking, no entry variants

### Danger (предупреждающие) - 219 объектов
- 11, 18-31: priority, danger, bends, construction, crossings

### Mandatory (предписывающие) - 163 объекта
- 33-40: go right/left/straight, keep right/left, roundabout

### Other (прочие) - 274 объекта
- 6, 12-14, 17, 32, 41-42: restriction ends, priority road, give way, stop

## Разбиение на train/val/test

- Train: 630 изображений (70%)
- Validation: 135 изображений (15%)
- Test: 135 изображений (15%)
- Random seed: 42
- Разбиение на уровне отдельных изображений (не последовательностей)

## Формат аннотаций YOLO

Каждый `.txt` файл содержит строки формата:
<class_id> <x_center> <y_center> <width> <height>
Все координаты нормализованы к размеру изображения (0-1).