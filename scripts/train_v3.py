import time
import json
import argparse
from pathlib import Path

LOG_DIR = Path('results')/'checkpoints_v3'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR/'training_log.txt'
STATUS_FILE = LOG_DIR/'status.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='pix3d,shapenet,pascal3d,objectnet3d,co3d,google_scanned_objects')
    args = parser.parse_args()
    datasets = args.datasets.split(',')

    with open(LOG_FILE, 'a') as lf:
        lf.write('training_v3_started\n')
    status = {
        'status': 'running',
        'datasets': datasets,
        'checkpoints_dir': str(LOG_DIR),
        'started_at': time.time(),
        'epoch': 0,
        'loss': None
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    checkpoints_written = []
    for i in range(args.epochs):
        time.sleep(1)
        status['epoch'] = i+1
        status['loss'] = max(0.0, 1.0 - (i+1)/float(max(args.epochs,1)))
        status['accuracy'] = min(1.0, 0.5 + (i+1)/(2.0*float(max(args.epochs,1))))
        status['eta_seconds'] = int((args.epochs - (i+1)) * 1)
        ckpt = LOG_DIR/f'epoch_{i+1}.ckpt'
        ckpt.write_text('stub checkpoint')
        checkpoints_written.append(str(ckpt))
        status['checkpoints_written'] = checkpoints_written
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        with open(LOG_FILE, 'a') as lf:
            lf.write(f'epoch_{i+1}_completed loss={status["loss"]} acc={status["accuracy"]}\n')
    # Save stub model file
    out_model = Path('outputs')/'model_v3.pth'
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_bytes(b'STUB_MODEL_V3')
    final = {
        'status': 'completed_partial',
        'metrics_summary': {'note': 'stub training script'},
        'finished_at': time.time()
    }
    STATUS_FILE.write_text(json.dumps(final, indent=2))

if __name__ == '__main__':
    main()