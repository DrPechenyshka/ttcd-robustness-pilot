# Воспроизводимость пилотного эксперимента

## Окружение
- Python: 3.11.9 (pyenv)
- PyTorch: 2.5.1+cu124
- Ultralytics: 8.3.40
- OpenCV: 4.10.0.84
- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
- ОС: Linux

## Установка
```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate.fish  # или . .venv/bin/activate для bash
pip install -r requirements.txt