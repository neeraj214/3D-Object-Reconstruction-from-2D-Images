import os
import json
import time
from pathlib import Path
import requests

BASE = os.environ.get('API_BASE', 'http://127.0.0.1:5000')
OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def check_url(url):
    try:
        r = requests.get(BASE + url, timeout=20)
        return r.status_code
    except Exception as e:
        return str(e)

def main():
    log = {}
    # 1. health
    r1 = requests.get(BASE + '/health', timeout=15)
    log['health'] = {'status_code': r1.status_code, 'json': (r1.json() if r1.headers.get('content-type','').startswith('application/json') else None)}
    # 2. datasets/list
    r2 = requests.get(BASE + '/datasets/list', timeout=30)
    d2 = r2.json() if r2.headers.get('content-type','').startswith('application/json') else {}
    log['datasets_list'] = {'status_code': r2.status_code, 'datasets_keys': list((d2.get('datasets') or {}).keys())}
    # 3. get_categories
    r3 = requests.get(BASE + '/get_categories', timeout=30)
    d3 = r3.json() if r3.headers.get('content-type','').startswith('application/json') else {}
    log['get_categories'] = {'status_code': r3.status_code, 'keys': list(d3.keys())}
    # 4. get_dataset_images
    r4 = requests.get(BASE + '/get_dataset_images', params={'dataset':'pix3d','category':'bed','max_items':3}, timeout=60)
    d4 = r4.json() if r4.headers.get('content-type','').startswith('application/json') else {}
    log['get_dataset_images'] = {'status_code': r4.status_code, 'count': len(d4.get('items') or [])}
    # 4b. datasets/category alias
    r4b = requests.get(BASE + '/datasets/category/bed', params={'dataset':'pix3d','max_items':3}, timeout=60)
    d4b = r4b.json() if r4b.headers.get('content-type','').startswith('application/json') else {}
    log['datasets_category'] = {'status_code': r4b.status_code, 'count': len(d4b.get('items') or [])}
    # 5. predict
    img_path = Path('data/pix3d/img/bed/0042.jpg')
    predict_res = None
    if img_path.exists():
        with open(img_path, 'rb') as f:
            pr = requests.post(BASE + '/predict', files={'file': f}, params={'n_points':20000}, timeout=300)
        pj = pr.json() if pr.headers.get('content-type','').startswith('application/json') else {}
        predict_res = {
            'status_code': pr.status_code,
            'has_output_urls': 'output_urls' in pj,
            'output_urls': pj.get('output_urls', {})
        }
        # reachability
        reach = {}
        for k,u in predict_res['output_urls'].items():
            if not u: 
                reach[k] = 'skip'
                continue
            reach[k] = check_url(u)
        predict_res['reachability'] = reach
    else:
        predict_res = {'status_code': 0, 'error': 'sample image missing at '+str(img_path)}
    log['predict'] = predict_res

    OUT_DIR.joinpath('validation_log.json').write_text(json.dumps(log, indent=2))

    # Write final status
    success = (log['health']['status_code']==200 and log['predict']['status_code'] in (200,201))
    status = {
        'success': bool(success),
        'timestamp': time.time(),
        'notes': 'Automated validation complete',
        'report_path': 'results/final_report.md'
    }
    OUT_DIR.joinpath('final_status.json').write_text(json.dumps(status, indent=2))

    # Generate a concise report now; full report will be enhanced by orchestrator
    lines = []
    lines.append('# Final Report (Automated Validation)')
    lines.append('')
    lines.append('## Endpoint Checks')
    for k,v in log.items():
        lines.append(f'- {k}: {json.dumps(v)}')
    OUT_DIR.joinpath('final_report.md').write_text('\n'.join(lines))

if __name__ == '__main__':
    main()