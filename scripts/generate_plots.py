import matplotlib.pyplot as plt
import numpy as np

# Настройка стиля для научной публикации
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.frameon': True,
    'legend.edgecolor': 'black'
})

# Данные из Таблицы 1
levels = ['Clean', 'Medium', 'High']
x = np.arange(len(levels))
width = 0.35

# Метрики
map50_baseline = [0.9793, 0.7707, 0.0108]
map50_proposed = [0.9691, 0.7491, 0.0125]

map5095_baseline = [0.8455, 0.6629, 0.0032]
map5095_proposed = [0.7212, 0.5306, 0.0030]

recall_baseline = [0.9695, 0.6985, 0.0313]
recall_proposed = [0.9063, 0.6869, 0.0330]

# ==========================================
# График 1: Сравнение mAP@0.5 по уровням искажений
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 5))
rects1 = ax1.bar(x - width/2, map50_baseline, width, label='Baseline', color='#1f77b4', edgecolor='black')
rects2 = ax1.bar(x + width/2, map50_proposed, width, label='Proposed (Усиленные аугм.)', color='#ff7f0e', edgecolor='black')

ax1.set_ylabel('mAP@0.5')
ax1.set_title('Влияние синтетических искажений на точность детекции (mAP@0.5)')
ax1.set_xticks(x)
ax1.set_xticklabels(levels)
ax1.legend(loc='lower left')
ax1.set_ylim(0, 1.1)

# Добавление значений на столбцы
for rect in rects1 + rects2:
    height = rect.get_height()
    ax1.annotate(f'{height:.3f}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/pilot/plot_map50_comparison.png', bbox_inches='tight')
print("График 1 сохранен: results/pilot/plot_map50_comparison.png")

# ==========================================
# График 2: "Цена улучшения" на Clean данных
# ==========================================
metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Recall@0.5']
baseline_vals = [0.9793, 0.8455, 0.9695]
proposed_vals = [0.9691, 0.7212, 0.9063]
drop = [b - p for b, p in zip(baseline_vals, proposed_vals)]

fig2, ax2 = plt.subplots(figsize=(8, 5))
x2 = np.arange(len(metrics))

# Столбцы Proposed
bars = ax2.bar(x2, proposed_vals, width=0.6, label='Proposed', color='#2ca02c', edgecolor='black')
# Линия Baseline для сравнения
ax2.plot(x2, baseline_vals, color='#d62728', marker='o', linewidth=2, markersize=8, label='Baseline')

ax2.set_ylabel('Значение метрики')
ax2.set_title('Деградация качества на "чистых" данных (Цена усиленных аугментаций)')
ax2.set_xticks(x2)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.set_ylim(0.6, 1.05)

# Аннотация падения
for i, (b, p, d) in enumerate(zip(baseline_vals, proposed_vals, drop)):
    ax2.annotate(f'↓ {d:.3f}', 
                 xy=(i, p), 
                 xytext=(0, -25), 
                 textcoords="offset points",
                 ha='center', va='top', color='#d62728', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('results/pilot/plot_clean_degradation.png', bbox_inches='tight')
print("График 2 сохранен: results/pilot/plot_clean_degradation.png")