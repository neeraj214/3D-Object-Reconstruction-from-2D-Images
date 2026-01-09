import json
from pathlib import Path
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prev', type=str, required=True)
    ap.add_argument('--curr', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    prev = json.loads(Path(args.prev).read_text()).get('aggregate', {})
    curr = json.loads(Path(args.curr).read_text()).get('aggregate', {})
    keys = set(list(prev.keys()) + list(curr.keys()))
    delta = {}
    for k in keys:
        try:
            delta[k] = (curr.get(k) or 0) - (prev.get(k) or 0)
        except Exception:
            pass
    out = {'prev': prev, 'current': curr, 'delta': delta}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()