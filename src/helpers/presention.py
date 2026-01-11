import matplotlib.pyplot as plt
import numpy as np

# Data
qualities = ['4K', '360p']
methods = ['My Software', 'Manual Photo']

# Values
ssim_values = [
    [0.76, 0.96],   # 4K
    [0.65, 0.85]    # 360p
]

ocr_values = [
    [0.88, 0.92],   # 4K
    [0.51, 0.81]    # 360p
]

x = np.arange(len(qualities))  # the label locations
width = 0.35  # the width of the bars

# ------------------------
# SSIM Chart
fig1, ax1 = plt.subplots(figsize=(8,5))

rects1 = ax1.bar(x - width/2, [ssim_values[0][0], ssim_values[1][0]], width, label='My Software')
rects2 = ax1.bar(x + width/2, [ssim_values[0][1], ssim_values[1][1]], width, label='Manual Photo')

ax1.set_ylabel('SSIM', fontsize=14)
ax1.set_title('SSIM Comparison', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(qualities, fontsize=13)
ax1.set_yticks(np.linspace(0, 1, 6))
ax1.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)], fontsize=12)
ax1.set_ylim(0,1.1)
ax1.legend(fontsize=12)

# Label bars
for rect in rects1 + rects2:
    height = rect.get_height()
    ax1.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0,5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12)

plt.tight_layout()
plt.show()

# ------------------------
# OCR Accuracy Chart
fig2, ax2 = plt.subplots(figsize=(8,5))

rects3 = ax2.bar(x - width/2, [ocr_values[0][0], ocr_values[1][0]], width, label='My Software')
rects4 = ax2.bar(x + width/2, [ocr_values[0][1], ocr_values[1][1]], width, label='Manual Photo')

ax2.set_ylabel('OCR Accuracy', fontsize=14)
ax2.set_title('OCR Accuracy Comparison', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(qualities, fontsize=13)
ax2.set_yticks(np.linspace(0, 1, 6))
ax2.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)], fontsize=12)
ax2.set_ylim(0,1.1)
ax2.legend(fontsize=12)

# Label bars
for rect in rects3 + rects4:
    height = rect.get_height()
    ax2.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0,5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12)

plt.tight_layout()
plt.show()
