#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="ttcd_robustness_pilot"

echo "[setup] Creating project structure for ${PROJECT_NAME}..."

# Создание директорий
mkdir -p "${PROJECT_NAME}"/{configs,scripts,docs}
mkdir -p "${PROJECT_NAME}"/src/{data,augmentations,evaluation,utils}
mkdir -p "${PROJECT_NAME}"/tests
mkdir -p "${PROJECT_NAME}"/data/raw/{images,annotations}
mkdir -p "${PROJECT_NAME}"/data/processed/{splits,pilot_subset}
mkdir -p "${PROJECT_NAME}"/results/pilot/{checkpoints,logs,metrics,visualizations}
mkdir -p "${PROJECT_NAME}"/results/main

# Создание __init__.py
for d in src src/data src/augmentations src/evaluation src/utils tests; do
    touch "${PROJECT_NAME}/${d}/__init__.py"
done

# .gitkeep для пустых директорий
touch "${PROJECT_NAME}/results/pilot/.gitkeep"

# .gitignore
cat > "${PROJECT_NAME}/.gitignore" << 'EOF'
# Данные (не коммитим)
data/raw/
data/processed/
results/pilot/checkpoints/
results/pilot/logs/
results/main/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
dist/
build/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# W&B / tensorboard
wandb/
runs/
EOF

echo "[setup] Done. Next: populate ${PROJECT_NAME}/ with source files."