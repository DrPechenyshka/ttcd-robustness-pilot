"""
Конвертация GTSDB аннотаций в COCO формат.
Формат GTSDB: #ImgNo#.ppm;#leftCol#;#topRow#;#rightCol#;#bottomRow#;#ClassID#
"""
import json
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import argparse


# Маппинг 43 классов GTSDB в 4 укрупнённые категории
# На основе ReadMe.txt
GTSDB_CLASS_MAPPING = {
    # Prohibitory (запрещающие) - круглые с красной каймой
    0: "prohibitory",   # speed limit 20
    1: "prohibitory",   # speed limit 30
    2: "prohibitory",   # speed limit 50
    3: "prohibitory",   # speed limit 60
    4: "prohibitory",   # speed limit 70
    5: "prohibitory",   # speed limit 80
    7: "prohibitory",   # speed limit 100
    8: "prohibitory",   # speed limit 120
    9: "prohibitory",   # no overtaking
    10: "prohibitory",  # no overtaking (trucks)
    15: "prohibitory",  # no traffic both ways
    16: "prohibitory",  # no trucks
    
    # Danger/Warning (предупреждающие) - треугольные
    11: "danger",       # priority at next intersection
    18: "danger",       # danger (general)
    19: "danger",       # bend left
    20: "danger",       # bend right
    21: "danger",       # bend
    22: "danger",       # uneven road
    23: "danger",       # slippery road
    24: "danger",       # road narrows
    25: "danger",       # construction
    26: "danger",       # traffic signal
    27: "danger",       # pedestrian crossing
    28: "danger",       # school crossing
    29: "danger",       # cycles crossing
    30: "danger",       # snow
    31: "danger",       # animals
    
    # Mandatory (предписывающие) - круглые синие
    33: "mandatory",    # go right
    34: "mandatory",    # go left
    35: "mandatory",    # go straight
    36: "mandatory",    # go right or straight
    37: "mandatory",    # go left or straight
    38: "mandatory",    # keep right
    39: "mandatory",    # keep left
    40: "mandatory",    # roundabout
    
    # Other (прочие)
    6: "other",         # restriction ends 80
    12: "other",        # priority road
    13: "other",        # give way
    14: "other",        # stop
    17: "other",        # no entry
    32: "other",        # restriction ends
    41: "other",        # restriction ends (overtaking)
    42: "other",        # restriction ends (overtaking trucks)
}

# ID категорий в COCO
CATEGORY_IDS = {
    "prohibitory": 1,
    "danger": 2,
    "mandatory": 3,
    "other": 4,
}


def parse_gtsdb_gt(gt_file: Path) -> list:
    """
    Парсит gt.txt в формате:
    #ImgNo#.ppm;#leftCol#;#topRow#;#rightCol#;#bottomRow#;#ClassID#
    """
    annotations = []
    
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(';')
            if len(parts) != 6:
                print(f"Warning: skipping malformed line: {line}")
                continue
            
            filename = parts[0]
            left_col = int(parts[1])
            top_row = int(parts[2])
            right_col = int(parts[3])
            bottom_row = int(parts[4])
            class_id = int(parts[5])
            
            # Конвертируем в COCO формат: [x, y, width, height]
            x = left_col
            y = top_row
            width = right_col - left_col
            height = bottom_row - top_row
            
            annotations.append({
                'filename': filename,
                'bbox': [x, y, width, height],
                'class_id': class_id,
            })
    
    return annotations


def create_coco_dataset(
    images_dir: Path,
    annotations: list,
    output_path: Path,
    dataset_name: str = "gtsdb"
):
    """Создаёт COCO-формат датасет."""
    
    # Группируем аннотации по изображениям
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['filename']].append(ann)
    
    # Получаем список всех PPM изображений
    image_files = sorted(images_dir.glob("*.ppm"))
    
    coco = {
        "info": {
            "description": dataset_name,
            "version": "1.0",
            "year": 2013,
            "contributor": "GTSDB Benchmark",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "prohibitory", "supercategory": "traffic_sign"},
            {"id": 2, "name": "danger", "supercategory": "traffic_sign"},
            {"id": 3, "name": "mandatory", "supercategory": "traffic_sign"},
            {"id": 4, "name": "other", "supercategory": "traffic_sign"},
        ],
    }
    
    ann_id = 1
    skipped_images = 0
    
    for img_id, img_file in enumerate(image_files, start=1):
        # Читаем изображение для получения размеров
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: cannot read {img_file}, skipping")
            skipped_images += 1
            continue
        
        height, width = img.shape[:2]
        
        coco["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": width,
            "height": height,
        })
        
        # Добавляем аннотации для этого изображения
        filename = img_file.name
        if filename in img_to_anns:
            for ann in img_to_anns[filename]:
                # Маппим класс
                class_id_gtsdb = ann['class_id']
                category_name = GTSDB_CLASS_MAPPING.get(class_id_gtsdb)
                
                if category_name is None:
                    print(f"Warning: unknown class_id {class_id_gtsdb} in {filename}")
                    continue
                
                category_id = CATEGORY_IDS[category_name]
                x, y, w, h = ann['bbox']
                
                # Проверка валидности bbox
                if w <= 0 or h <= 0:
                    print(f"Warning: invalid bbox in {filename}: {ann['bbox']}")
                    continue
                
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1
    
    # Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Created COCO dataset: {output_path}")
    print(f"{'='*60}")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    if skipped_images > 0:
        print(f"  Skipped images: {skipped_images}")
    
    # Статистика по классам
    class_counts = defaultdict(int)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        cat_name = next(c["name"] for c in coco["categories"] if c["id"] == cat_id)
        class_counts[cat_name] += 1
    
    print(f"\n  Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls:15s}: {count:4d}")
    
    # Статистика по изображениям с аннотациями
    images_with_anns = len(set(ann["image_id"] for ann in coco["annotations"]))
    print(f"\n  Images with annotations: {images_with_anns} / {len(coco['images'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GTSDB annotations to COCO format"
    )
    parser.add_argument(
        "--gt-file", 
        type=Path, 
        default=Path("data/raw/gtsdb/gt.txt"),
        help="Path to gt.txt (default: data/raw/gtsdb/gt.txt)"
    )
    parser.add_argument(
        "--images-dir", 
        type=Path, 
        default=Path("data/raw/gtsdb"),
        help="Path to directory with PPM images (default: data/raw/gtsdb)"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/processed/gtsdb_coco.json"),
        help="Output COCO JSON path (default: data/processed/gtsdb_coco.json)"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="gtsdb",
        help="Dataset name for COCO info"
    )
    
    args = parser.parse_args()
    
    # Проверка существования файлов
    if not args.gt_file.exists():
        print(f"Error: gt.txt not found at {args.gt_file}")
        print("Please extract GTSDB archive to data/raw/gtsdb/")
        return
    
    if not args.images_dir.exists():
        print(f"Error: images directory not found at {args.images_dir}")
        return
    
    # Парсим аннотации
    print(f"Parsing GTSDB ground truth from {args.gt_file}...")
    annotations = parse_gtsdb_gt(args.gt_file)
    print(f"Parsed {len(annotations)} annotations")
    
    # Создаём COCO датасет
    create_coco_dataset(
        args.images_dir, 
        annotations, 
        args.output, 
        args.dataset_name
    )


if __name__ == "__main__":
    main()