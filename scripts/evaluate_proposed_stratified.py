"""
Стратифицированная оценка робастности PROPOSED модели.
"""
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json

def apply_corruption(img_path, output_path, severity):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    rng = np.random.default_rng(42)
    
    if severity == 1:  # Medium
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        for _ in range(rng.integers(1, 3)):
            pts = rng.integers(0, min(h, w), size=(rng.integers(4, 7), 2))
            hull = cv2.convexHull(pts)
            color = (int(rng.integers(50, 150)), int(rng.integers(50, 150)), int(rng.integers(50, 150)))
            cv2.fillPoly(img, [hull], color)
            
    elif severity == 2:  # High
        img = cv2.GaussianBlur(img, (9, 9), 3.0)
        for _ in range(rng.integers(2, 5)):
            pts = rng.integers(0, min(h, w), size=(rng.integers(5, 10), 2))
            hull = cv2.convexHull(pts)
            color = (int(rng.integers(30, 100)), int(rng.integers(30, 100)), int(rng.integers(30, 100)))
            cv2.fillPoly(img, [hull], color)
        max_offset = int(min(w, h) * 0.1)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = src_pts + rng.integers(-max_offset, max_offset, size=src_pts.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (w, h))

    cv2.imwrite(str(output_path), img)
    return True

def get_scalar(val):
    if isinstance(val, list):
        return float(sum(val) / len(val)) if len(val) > 0 else 0.0
    if isinstance(val, np.ndarray):
        return float(val.mean()) if val.size > 0 else 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

def evaluate_proposed():
    print("="*70)
    print("СТРАТИФИЦИРОВАННАЯ ОЦЕНКА ROBUSTNESS (PROPOSED)")
    print("="*70)
    
    # !!! ИЗМЕНЕН ПУТЬ К МОДЕЛИ !!!
    model_path = "results/pilot/proposed/weights/best.pt"
    test_images_dir = Path("data/processed/gtsdb_yolo/test/images")
    test_labels_dir = Path("data/processed/gtsdb_yolo/test/labels")
    
    print(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    severity_levels = {
        0: "Clean (Без искажений)",
        1: "Medium (Средние искажения)",
        2: "High (Сильные искажения)"
    }
    
    results_summary = []
    
    for severity, name in severity_levels.items():
        print(f"\n[{severity}] Оценка уровня: {name}")
        
        temp_test_dir = Path(f"data/processed/temp_test_proposed_sev_{severity}")
        temp_images_dir = temp_test_dir / "images"
        temp_labels_dir = temp_test_dir / "labels"
        
        if temp_test_dir.exists():
            shutil.rmtree(temp_test_dir)
        
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for label_file in test_labels_dir.glob("*.txt"):
            shutil.copy(label_file, temp_labels_dir / label_file.name)
        
        print("  Применение искажений к тестовым изображениям...")
        for img_file in test_images_dir.glob("*.jpg"):
            out_path = temp_images_dir / img_file.name
            apply_corruption(img_file, out_path, severity)
        
        temp_data_yaml = temp_test_dir / "data.yaml"
        with open(temp_data_yaml, 'w') as f:
            f.write(f"path: {temp_test_dir.absolute()}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            f.write("nc: 4\n")
            f.write("names:\n  0: prohibitory\n  1: danger\n  2: mandatory\n  3: other\n")
        
        print("  Запуск оценки...")
        metrics = model.val(
            data=str(temp_data_yaml),
            split="val",
            imgsz=640,
            batch=8,
            verbose=False,
            plots=False
        )
        
        map50 = get_scalar(metrics.box.map50)
        map50_95 = get_scalar(metrics.box.map)
        recall = get_scalar(metrics.box.r)
        
        print(f"  -> mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Recall: {recall:.4f}")
        
        results_summary.append({
            "Level": name,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "Recall": recall
        })
        
        shutil.rmtree(temp_test_dir)
    
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (PROPOSED)")
    print("="*70)
    print(f"{'Уровень искажений':<30} | {'mAP50':<8} | {'mAP50-95':<8} | {'Recall':<8}")
    print("-" * 70)
    for res in results_summary:
        print(f"{res['Level']:<30} | {res['mAP50']:<8.4f} | {res['mAP50-95']:<8.4f} | {res['Recall']:<8.4f}")
    print("="*70)
    
    results_dir = Path("results/pilot/proposed")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "stratified_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Результаты сохранены в {results_dir / 'stratified_results.json'}")

if __name__ == "__main__":
    evaluate_proposed()