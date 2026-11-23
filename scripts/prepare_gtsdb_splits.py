"""
Разбиение GTSDB на train/val/test с конвертацией PPM в JPG и сохранением в формате YOLO.
"""
import json
from pathlib import Path
import random
import cv2

def prepare_splits():
    # Пути
    coco_path = Path("data/processed/gtsdb_coco.json")
    images_dir = Path("data/raw/gtsdb/FullIJCNN2013")
    output_dir = Path("data/processed/gtsdb_yolo")
    
    # Создаем структуру
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Загружаем COCO аннотации
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    # Создаем маппинг image_id -> filename
    img_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    
    # Группируем аннотации по image_id
    img_id_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # Получаем список всех ID изображений и перемешиваем (с фиксированным seed для воспроизводимости)
    all_img_ids = list(img_id_to_filename.keys())
    random.seed(42)
    random.shuffle(all_img_ids)
    
    # Разбиение 70% / 15% / 15%
    n_total = len(all_img_ids)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    train_ids = all_img_ids[:n_train]
    val_ids = all_img_ids[n_train:n_train+n_val]
    test_ids = all_img_ids[n_train+n_val:]
    
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    
    # Карта категорий COCO -> YOLO (0-indexed)
    # COCO: prohibitory=1, danger=2, mandatory=3, other=4
    # YOLO: prohibitory=0, danger=1, mandatory=2, other=3
    coco_to_yolo = {1: 0, 2: 1, 3: 2, 4: 3}
    
    processed_count = 0
    for split_name, img_ids in splits.items():
        for img_id in img_ids:
            filename = img_id_to_filename[img_id]  # например, "00000.ppm"
            src_img_path = images_dir / filename
            
            # Читаем изображение через OpenCV (поддерживает .ppm)
            img = cv2.imread(str(src_img_path))
            if img is None:
                print(f"Warning: cannot read {src_img_path}")
                continue
                
            h, w = img.shape[:2]
            
            # Конвертируем и сохраняем как .jpg для совместимости с YOLOv8
            new_filename = Path(filename).stem + ".jpg"
            dst_img_path = output_dir / split_name / "images" / new_filename
            cv2.imwrite(str(dst_img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Создаем YOLO label файл
            label_filename = Path(new_filename).stem + ".txt"
            dst_label_path = output_dir / split_name / "labels" / label_filename
            
            with open(dst_label_path, 'w') as f:
                for ann in img_id_to_anns.get(img_id, []):
                    yolo_class = coco_to_yolo[ann["category_id"]]
                    x, y, box_w, box_h = ann["bbox"]
                    
                    # Нормализация по YOLO: x_center, y_center, width, height (от 0 до 1)
                    x_center = (x + box_w / 2) / w
                    y_center = (y + box_h / 2) / h
                    norm_w = box_w / w
                    norm_h = box_h / h
                    
                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            processed_count += 1
    
    print(f"Split created successfully in {output_dir}")
    print(f"  Processed images: {processed_count}")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val: {len(val_ids)} images")
    print(f"  Test: {len(test_ids)} images")

if __name__ == "__main__":
    prepare_splits()