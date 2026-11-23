"""
Стратифицированная оценка робастности модели на тестовом наборе с разными уровнями искажений.
"""
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random
import json


def apply_corruption(img_path, output_path, severity):
    """
    Применяет синтетические искажения к изображению.
    severity: 0 (Clean), 1 (Medium), 2 (High)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    rng = np.random.default_rng(42)  # Фиксированный seed для воспроизводимости
    
    if severity == 1:  # Medium
        # 1. Размытие
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        # 2. Легкая полигональная окклюзия (1-2 полигона, 20-40% площади)
        for _ in range(rng.integers(1, 3)):
            pts = rng.integers(0, min(h, w), size=(rng.integers(4, 7), 2))
            hull = cv2.convexHull(pts)
            # Явное преобразование numpy.int64 -> Python int
            color = (int(rng.integers(50, 150)),
                     int(rng.integers(50, 150)),
                     int(rng.integers(50, 150)))
            cv2.fillPoly(img, [hull], color)
            
    elif severity == 2:  # High
        # 1. Сильное размытие
        img = cv2.GaussianBlur(img, (9, 9), 3.0)
        # 2. Сильная полигональная окклюзия (2-4 полигона, 40-70% площади)
        for _ in range(rng.integers(2, 5)):
            pts = rng.integers(0, min(h, w), size=(rng.integers(5, 10), 2))
            hull = cv2.convexHull(pts)
            # Явное преобразование numpy.int64 -> Python int
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

    cv2.imwrite(str(output_path), img)
    return True


def get_scalar(val):
    """
    Универсальное извлечение скаляра из метрик Ultralytics.
    Обрабатывает: numpy.ndarray, list, numpy scalar, Python float.
    """
    # Список (случай полного провала модели)
    if isinstance(val, list):
        if len(val) == 0:
            return 0.0
        return float(sum(val) / len(val))
    # NumPy array
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return 0.0
        return float(val.mean())
    # NumPy scalar или Python float
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def evaluate_stratified():
    print("=" * 70)
    print("СТРАТИФИЦИРОВАННАЯ ОЦЕНКА ROBUSTNESS (BASELINE)")
    print("=" * 70)
    
    model_path = "results/pilot/baseline/weights/best.pt"
    test_images_dir = Path("data/processed/gtsdb_yolo/test/images")
    test_labels_dir = Path("data/processed/gtsdb_yolo/test/labels")
    
    # Загрузка модели
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
        
        # Создаем временную директорию для этого уровня искажений
        temp_test_dir = Path(f"data/processed/temp_test_sev_{severity}")
        temp_images_dir = temp_test_dir / "images"
        temp_labels_dir = temp_test_dir / "labels"
        
        # Удаляем старую временную директорию, если она есть
        if temp_test_dir.exists():
            shutil.rmtree(temp_test_dir)
        
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Копируем лейблы (они не меняются)
        for label_file in test_labels_dir.glob("*.txt"):
            shutil.copy(label_file, temp_labels_dir / label_file.name)
        
        # Применяем искажения к изображениям
        print("  Применение искажений к тестовым изображениям...")
        for img_file in test_images_dir.glob("*.jpg"):
            out_path = temp_images_dir / img_file.name
            apply_corruption(img_file, out_path, severity)
        
        # Создаем временный data.yaml для этого уровня
        temp_data_yaml = temp_test_dir / "data.yaml"
        with open(temp_data_yaml, 'w') as f:
            f.write(f"path: {temp_test_dir.absolute()}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            f.write("nc: 4\n")
            f.write("names:\n  0: prohibitory\n  1: danger\n  2: mandatory\n  3: other\n")
        
        # Запуск валидации YOLO
        print("  Запуск оценки...")
        metrics = model.val(
            data=str(temp_data_yaml),
            split="val",
            imgsz=640,
            batch=8,
            verbose=False,
            plots=False
        )
        
        # === БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ МЕТРИК ===
        # Используем только универсальную функцию get_scalar
        map50 = get_scalar(metrics.box.map50)
        map50_95 = get_scalar(metrics.box.map)
        recall = get_scalar(metrics.box.r)  # macro-average по 4 классам
        # =====================================
        
        print(f"  -> mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Recall: {recall:.4f}")
        
        results_summary.append({
            "Level": name,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "Recall": recall
        })
        
        # Очистка временной директории
        shutil.rmtree(temp_test_dir)
    
    # Вывод итоговой таблицы
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (BASELINE)")
    print("=" * 70)
    print(f"{'Уровень искажений':<30} | {'mAP50':<8} | {'mAP50-95':<8} | {'Recall':<8}")
    print("-" * 70)
    for res in results_summary:
        print(f"{res['Level']:<30} | {res['mAP50']:<8.4f} | {res['mAP50-95']:<8.4f} | {res['Recall']:<8.4f}")
    print("=" * 70)
    
    # Сохранение результатов в файл
    results_dir = Path("results/pilot/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "stratified_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Результаты сохранены в {results_dir / 'stratified_results.json'}")


if __name__ == "__main__":
    evaluate_stratified()