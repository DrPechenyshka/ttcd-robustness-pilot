"""
Запуск пилотного обучения Proposed с усиленными геометрическими аугментациями.
"""
from ultralytics import YOLO
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def run_pilot_proposed():
    print("="*60)
    print("ЗАПУСК ПИЛОТНОГО ОБУЧЕНИЯ: PROPOSED (УСИЛЕННЫЕ АУГМЕНТАЦИИ)")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ Обнаружен GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️ ВНИМАНИЕ: CUDA не доступна.")
        device = 'cpu'

    print("Загрузка модели yolov8m.pt...")
    model = YOLO('yolov8m.pt')

    print("Начало обучения с усиленными аугментациями...")
    results = model.train(
        data='data/processed/gtsdb_yolo/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        device=device,
        project='results/pilot',
        name='proposed',
        exist_ok=True,  # Перезаписывает старую папку proposed
        patience=10,
        seed=42,
        workers=2,
        
        # === УСИЛЕННЫЕ АУГМЕНТАЦИИ (имитация corruption-aware) ===
        # Эти параметры ПЕРЕОПРЕДЕЛЯТ дефолтные значения Ultralytics
        degrees=10.0,         # Поворот до 10 градусов (было 0.0)
        translate=0.2,        # Сдвиг до 20% (было 0.1)
        scale=0.7,            # Масштабирование до 70% (было 0.5)
        shear=5.0,            # Сдвиг углов до 5 градусов (было 0.0)
        perspective=0.001,    # Легкая перспектива (было 0.0)
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,            # Mixup 10% (было 0.0)
        copy_paste=0.1,       # Copy-paste 10% (было 0.0)
    )
    
    print("="*60)
    print("✅ ПИЛОТНОЕ ОБУЧЕНИЕ PROPOSED ЗАВЕРШЕНО!")
    print(f"Результаты и веса сохранены в: results/pilot/proposed")
    print("="*60)

if __name__ == '__main__':
    run_pilot_proposed()