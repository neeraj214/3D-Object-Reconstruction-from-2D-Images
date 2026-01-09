import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path('results')
PPT_DIR = Path('ppt_images/new')

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PPT_DIR.mkdir(parents=True, exist_ok=True)

def read_metrics():
    metrics_csv = RESULTS_DIR/'metrics.csv'
    rows = []
    if metrics_csv.exists():
        for line in metrics_csv.read_text().splitlines():
            if not line or line.startswith('dataset'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                rows.append({'dataset': parts[0], 'cd': parts[1], 'fscore': parts[2]})
    return rows

def write_metrics_report(rows):
    md = [
        '# Metrics Report',
        '',
        '| Dataset | Chamfer Distance | F-score |',
        '|---|---:|---:|'
    ]
    if rows:
        for r in rows:
            md.append(f"| {r['dataset']} | {r['cd']} | {r['fscore']} |")
    else:
        md.append('| Pix3D | – | – |')
        md.append('| ShapeNet | – | – |')
        md.append('| Pascal3D+ | – | – |')
        md.append('| ObjectNet3D | – | – |')
        md.append('| CO3D | – | – |')
        md.append('| Google Scanned Objects | – | – |')
    (RESULTS_DIR/'metrics_report.md').write_text('\n'.join(md))

def write_dataset_summary():
    datasets = ['pix3d','shapenet','pascal3d','objectnet3d','co3d','google_scanned']
    lines = ['# Dataset Summary','']
    for d in datasets:
        info = Path('data')/d/'dataset_info.json'
        if info.exists():
            lines.append(f"- {d}: registered; info found")
        else:
            lines.append(f"- {d}: structure prepared; info missing or manual download required")
    (RESULTS_DIR/'dataset_summary.md').write_text('\n'.join(lines))

def draw_box(ax, xy, text):
    ax.add_patch(plt.Rectangle(xy, 1.8, 0.8, edgecolor='black', facecolor='#e8eef9'))
    ax.text(xy[0]+0.9, xy[1]+0.4, text, ha='center', va='center', fontsize=10)

def draw_arrow(ax, xy_from, xy_to):
    ax.annotate('', xy=xy_to, xytext=xy_from, arrowprops=dict(arrowstyle='->'))

def save_diagram(filename, boxes, arrows, title):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_axis_off()
    ax.set_xlim(0,10)
    ax.set_ylim(0,5)
    ax.text(5,4.7,title,ha='center',fontsize=12)
    for b in boxes:
        draw_box(ax, b['xy'], b['text'])
    for a in arrows:
        draw_arrow(ax, a['from'], a['to'])
    out = PPT_DIR/filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_images():
    save_diagram('model_architecture_diagram.png',
        boxes=[
            {'xy': (0.5,3.2), 'text': 'Input Image'},
            {'xy': (3.0,3.2), 'text': 'DPT Hybrid\n(attention + multi-scale)'},
            {'xy': (5.5,3.2), 'text': 'Depth Map'},
            {'xy': (8.0,3.2), 'text': 'Point Cloud'}
        ],
        arrows=[{'from': (2.3,3.6), 'to': (3.0,3.6)}, {'from': (4.8,3.6), 'to': (5.5,3.6)}, {'from': (7.3,3.6), 'to': (8.0,3.6)}],
        title='Model Architecture'
    )
    save_diagram('training_flowchart.png',
        boxes=[
            {'xy': (0.5,2.8), 'text': 'Load Datasets'},
            {'xy': (3.0,2.8), 'text': 'Preprocess'},
            {'xy': (5.5,2.8), 'text': 'Train Depth'},
            {'xy': (8.0,2.8), 'text': 'Validate & Save\nbest_model.pth'}
        ],
        arrows=[{'from': (2.3,3.2), 'to': (3.0,3.2)}, {'from': (4.8,3.2), 'to': (5.5,3.2)}, {'from': (7.3,3.2), 'to': (8.0,3.2)}],
        title='Training Flow'
    )
    save_diagram('inference_pipeline.png',
        boxes=[
            {'xy': (0.5,2.8), 'text': 'Input Image'},
            {'xy': (3.0,2.8), 'text': 'Depth Prediction'},
            {'xy': (5.5,2.8), 'text': 'Depth→PointCloud'},
            {'xy': (8.0,2.8), 'text': 'Visualization'}
        ],
        arrows=[{'from': (2.3,3.2), 'to': (3.0,3.2)}, {'from': (4.8,3.2), 'to': (5.5,3.2)}, {'from': (7.3,3.2), 'to': (8.0,3.2)}],
        title='Inference Pipeline'
    )

def write_bleu_style_table(rows):
    md = [
        '# BLEU-style Evaluation Table',
        '',
        '| Dataset | Score-1 | Score-2 | Score-3 |',
        '|---|---:|---:|---:|'
    ]
    if rows:
        for r in rows:
            md.append(f"| {r['dataset']} | {r.get('score1','-')} | {r.get('score2','-')} | {r.get('score3','-')} |")
    else:
        md.append('| Pix3D | - | - | - |')
        md.append('| ShapeNet | - | - | - |')
        md.append('| Pascal3D+ | - | - | - |')
        md.append('| ObjectNet3D | - | - | - |')
        md.append('| CO3D | - | - | - |')
        md.append('| Google Scanned Objects | - | - | - |')
    (RESULTS_DIR/'bleu_table.md').write_text('\n'.join(md))

def write_accuracy_table(rows):
    md = [
        '# Accuracy / Performance Comparison',
        '',
        '| Dataset | Depth MAE | CD | F-score | Throughput |',
        '|---|---:|---:|---:|---:|'
    ]
    if rows:
        for r in rows:
            md.append(f"| {r['dataset']} | {r.get('mae','-')} | {r.get('cd','-')} | {r.get('fscore','-')} | {r.get('fps','-')} |")
    else:
        md.append('| Pix3D | - | - | - | - |')
        md.append('| ShapeNet | - | - | - | - |')
        md.append('| Pascal3D+ | - | - | - | - |')
        md.append('| ObjectNet3D | - | - | - | - |')
        md.append('| CO3D | - | - | - | - |')
        md.append('| Google Scanned Objects | - | - | - | - |')
    (RESULTS_DIR/'accuracy_table.md').write_text('\n'.join(md))

def main():
    ensure_dirs()
    rows = read_metrics()
    write_metrics_report(rows)
    write_dataset_summary()
    generate_images()
    write_bleu_style_table(rows)
    write_accuracy_table(rows)
    print('reports and diagrams generated under results/ and ppt_images/new/')

if __name__ == '__main__':
    main()