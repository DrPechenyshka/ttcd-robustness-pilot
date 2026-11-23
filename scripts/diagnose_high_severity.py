"""
Диагностика нулевых результатов на уровне High.
Цель: понять причину отказа моделей (реальный провал vs ошибка трансформации).
"""
import cv2
import numpy as np
from pathlib import Path
import shutil
from ultralytics import YOLO


def apply_high_corruption(img, rng):
    """Применяет искажения уровня High (в точности как в evaluate_stratified.py)"""
    h, w = img.shape[:2]
    
    # 1. Сильное размытие
    img = cv2.GaussianBlur(img, (9, 9), 3.0)
    
    # 2. Сильная полигональная окклюзия
    for _ in range(rng.integers(2, 5)):
        pts = rng.integers(0, min(h, w), size=(rng.integers(5, 10), 2))
        hull = cv2.convexHull(pts)
        color = (int(rng.integers(30, 100)),
                 int(rng.integers(30, 100)),
                 int(rng.integers(30, 100)))
        cv2.fillPoly(img, [hull], color)
    
    # 3. Перспективное искажение
    max_offset = int(min(w, h) * 0.1)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = src_pts + rng.integers(-max_offset, max_offset, size=src_pts.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.warpPerspective(img, M, (w, h))
    
    return img, M


def transform_bbox(bbox, M, img_w, img_h):
    """Трансформирует bbox (x, y, w, h) через матрицу гомографии M."""
    x, y, bw, bh = bbox
    pts = np.float32([
        [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
    ]).reshape(-1, 1, 2)
    
    pts_aug = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    
    x_new, y_new = pts_aug.min(axis=0)
    x2_new, y2_new = pts_aug.max(axis=0)
    
    # Обрезка по границам изображения
    x_new = max(0, x_new)
    y_new = max(0, y_new)
    x2_new = min(img_w, x2_new)
    y2_new = min(img_h, y2_new)
    
    new_w = x2_new - x_new
    new_h = y2_new - y_new
    
    return (x_new, y_new, new_w, new_h)


def diagnose():
    print("=" * 70)
    print("ДИАГНОСТИКА УРОВНЯ HIGH")
    print("=" * 70)
    
    model_path = "results/pilot/baseline/weights/best.pt"
    test_images_dir = Path("data/processed/gtsdb_yolo/test/images")
    test_labels_dir = Path("data/processed/gtsdb_yolo/test/labels")
    output_dir = Path("results/pilot/diagnostics/high")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(model_path)
    
    # Статистика
    total_gt = 0
    valid_gt_after_transform = 0
    zero_area_gt = 0
    total_detections = 0
    
    # Берём первые 10 изображений для детальной диагностики
    test_images = sorted(test_images_dir.glob("*.jpg"))[:10]
    
    for idx, img_path in enumerate(test_images):
        print(f"\n--- Изображение {idx+1}/{len(test_images)}: {img_path.name} ---")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("  Ошибка чтения изображения!")
            continue
            
        h, w = img.shape[:2]
        
        # Загружаем GT
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_bboxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    # Конвертируем нормализованные YOLO координаты в пиксели (x, y, w, h)
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    bw_px = bw * w
                    bh_px = bh * h
                    gt_bboxes.append((x, y, bw_px, bh_px))
        
        print(f"  Исходных GT-объектов: {len(gt_bboxes)}")
        total_gt += len(gt_bboxes)
        
        # Применяем искажения
        rng = np.random.default_rng(42 + idx)  # фиксированный seed для воспроизводимости
        img_corrupted, M = apply_high_corruption(img.copy(), rng)
        
        # Трансформируем GT и считаем валидные
        for bbox in gt_bboxes:
            new_bbox = transform_bbox(bbox, M, w, h)
            if new_bbox[2] <= 1.0 or new_bbox[3] <= 1.0: # Площадь схлопнулась
                zero_area_gt += 1
            else:
                valid_gt_after_transform += 1
        
        print(f"  Валидных GT после перспективы: {valid_gt_after_transform}")
        print(f"  GT с нулевой/микроскопической площадью: {zero_area_gt}")
        
        # Инференс модели
        results = model(img_corrupted, imgsz=640, conf=0.5, verbose=False)
        detections = results[0].boxes
        n_det = len(detections)
        total_detections += n_det
        print(f"  Детекций моделью (conf >= 0.5): {n_det}")
        
        # === Визуализация ===
        viz_img = img_corrupted.copy()
        
        # Рисуем трансформированный GT (зеленый)
        for bbox in gt_bboxes:
            new_bbox = transform_bbox(bbox, M, w, h)
            if new_bbox[2] > 1.0 and new_bbox[3] > 1.0:
                x, y, bw, bh = [int(v) for v in new_bbox]
                cv2.rectangle(viz_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
        # Рисуем детекции модели (красный)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[0].tolist()]
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Добавим текст с confidence
            cv2.putText(viz_img, f"{det.conf[0]:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Сохраняем
        viz_path = output_dir / f"{img_path.stem}_diag.jpg"
        cv2.imwrite(str(viz_path), viz_img)
        print(f"  Визуализация сохранена: {viz_path}")
    
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СТАТИСТИКА (по 10 изображениям)")
    print("=" * 70)
    print(f"  Всего исходных GT-объектов: {total_gt}")
    print(f"  Валидных после трансформации: {valid_gt_after_transform}")
    print(f"  Схлопнувшихся (площадь <= 1): {zero_area_gt}")
    print(f"  Всего детекций моделью: {total_detections}")
    print(f"\n  Визуализации сохранены в: {output_dir.absolute()}")
    print("  Откройте их и сравните: есть ли зеленые рамки (GT) и красные рамки (детекции).")


if __name__ == "__main__":
    diagnose()