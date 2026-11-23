"""
Запуск пилотного обучения Baseline на GTSDB.
"""
from ultralytics import YOLO
import torch
import sys
from pathlib import Path

# Добавляем корень проекта в путь, если понадобится
sys.path.append(str(Path(__file__).parent.parent))

def run_pilot_baseline():
    print("="*60)
    print("ЗАПУСК ПИЛОТНОГО ОБУЧЕНИЯ: BASELINE")
    print("="*60)
    
    # Проверка GPU
    if torch.cuda.is_available():
        print(f"✅ Обнаружен GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️ ВНИМАНИЕ: CUDA не доступна. Обучение на CPU будет очень медленным.")
        device = 'cpu'

    # Загрузка предобученной модели YOLOv8m
    print("Загрузка модели yolov8m.pt...")
    model = YOLO('yolov8m.pt')

    # Параметры обучения (согласно нашему дизайну эксперимента)
    print("Начало обучения...")
    results = model.train(
        data='data/processed/gtsdb_yolo/data.yaml',
        epochs=30,          # Пилотный прогон (не 100, чтобы быстро проверить пайплайн)
        imgsz=640,
        batch=8,           # Уменьшите до 8, если не хватает памяти GPU
        device=device,
        project='results/pilot',
        name='baseline',
        exist_ok=True,
        patience=10,        # Early stopping
        seed=42,
        workers=2,
        amp = True
        # Стандартные аугментации YOLOv8 включены по умолчанию (flip, hsv, mosaic и т.д.)
    )
    
    print("="*60)
    print("✅ ПИЛОТНОЕ ОБУЧЕНИЕ BASELINE ЗАВЕРШЕНО!")
    print(f"Результаты и веса сохранены в: results/pilot/baseline")
    print("="*60)

if __name__ == '__main__':
    run_pilot_baseline()