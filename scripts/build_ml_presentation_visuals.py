from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

OUT = Path('results') / 'presentation' / 'ml_dl_viz'
OUT.mkdir(parents=True, exist_ok=True)

def viz_architecture():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    boxes = []
    boxes.append(('Input', (0.05, 0.35, 0.12, 0.3)))
    boxes.append(('Conv + ReLU', (0.22, 0.45, 0.12, 0.2)))
    boxes.append(('Pool', (0.36, 0.5, 0.08, 0.1)))
    boxes.append(('Conv + ReLU', (0.48, 0.45, 0.12, 0.2)))
    boxes.append(('Pool', (0.62, 0.5, 0.08, 0.1)))
    boxes.append(('FC', (0.76, 0.45, 0.08, 0.2)))
    boxes.append(('Softmax', (0.88, 0.45, 0.09, 0.2)))
    for label, (x, y, w, h) in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', linewidth=1.5, edgecolor='#111', facecolor='#e5e7eb')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=11)
    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=1.5))
    arrow(0.17, 0.5, 0.22, 0.55)
    arrow(0.34, 0.55, 0.36, 0.55)
    arrow(0.44, 0.55, 0.48, 0.55)
    arrow(0.60, 0.55, 0.62, 0.55)
    arrow(0.70, 0.55, 0.76, 0.55)
    arrow(0.84, 0.55, 0.88, 0.55)
    ax.set_title('CNN Architecture for Image Classification', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'cnn_architecture.png', dpi=200)
    plt.close(fig)

def viz_loss_curves():
    epochs = np.arange(1, 51)
    train_loss = 1.5 * np.exp(-epochs / 18.0) + 0.02 * np.random.randn(50)
    val_loss = 1.6 * np.exp(-epochs / 25.0) + 0.02 * np.random.randn(50)
    val_loss[40:] += np.linspace(0.0, 0.12, 10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label='Training Loss', color='#2563eb', lw=2)
    ax.plot(epochs, val_loss, label='Validation Loss', color='#ef4444', lw=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss (50 epochs)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'loss_curves.png', dpi=200)
    plt.close(fig)

def viz_model_comparison():
    models = ['Your Model', 'Baseline B', 'Baseline A']
    accuracy = np.array([0.93, 0.88, 0.82])
    f1 = np.array([0.92, 0.86, 0.80])
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, accuracy, width=w, label='Accuracy', color='#10b981')
    ax.bar(x + w/2, f1, width=w, label='F1-Score', color='#6366f1')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Accuracy and F1-Score')
    for i in range(len(models)):
        ax.text(x[i] - w/2, accuracy[i] + 0.02, f'{accuracy[i]:.2f}', ha='center', fontsize=9)
        ax.text(x[i] + w/2, f1[i] + 0.02, f'{f1[i]:.2f}', ha='center', fontsize=9)
    ax.legend()
    ax.grid(alpha=0.15, axis='y')
    fig.tight_layout()
    fig.savefig(OUT / 'model_comparison.png', dpi=200)
    plt.close(fig)

def viz_confusion_matrix():
    labels = ['Negative', 'Positive']
    cm = np.array([[460, 40],[35, 465]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    ax.set_title('Confusion Matrix (Binary Classification)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / 'confusion_matrix.png', dpi=200)
    plt.close(fig)

def viz_distribution():
    rng = np.random.default_rng(42)
    data = rng.lognormal(mean=3.2, sigma=0.4, size=5000)
    data = np.clip(data, 10, 90)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=20, color='#f59e0b', edgecolor='#111', alpha=0.85)
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Data Distribution: Age')
    ax.grid(alpha=0.15, axis='y')
    fig.tight_layout()
    fig.savefig(OUT / 'data_distribution_histogram.png', dpi=200)
    plt.close(fig)

def main():
    viz_architecture()
    viz_loss_curves()
    viz_model_comparison()
    viz_confusion_matrix()
    viz_distribution()
    print(f'Visualizations saved to {OUT}')

if __name__ == '__main__':
    main()

