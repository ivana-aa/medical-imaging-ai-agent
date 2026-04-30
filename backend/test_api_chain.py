# -*- coding: utf-8 -*-
import urllib.request, json, time

t0 = time.time()
req = urllib.request.Request(
    'http://localhost:8001/api/demo-analyze',
    method='POST',
    headers={'Content-Length': '0'}
)
try:
    resp = urllib.request.urlopen(req, timeout=90)
    elapsed = time.time() - t0
    data = json.loads(resp.read().decode('utf-8'))

    with open('backend/api_result.json', 'w', encoding='utf-8') as f:
        json.dump({
            'elapsed': elapsed,
            'engine': data.get('engine'),
            'ai_keys': list(data.get('ai_result', {}).keys()),
            'det_keys': list(data.get('detection', {}).keys()),
            'overall': str(data.get('detection', {}).get('overall_assessment', ''))[:100],
            'clinical': data.get('detection', {}).get('clinical_recommendation', 'N/A'),
            'normal': data.get('detection', {}).get('normal_summary', [])[:2]
        }, f, ensure_ascii=False, indent=2)
    print('OK, elapsed=%.1fs' % elapsed)
except Exception as e:
    with open('backend/api_error.txt', 'w') as f:
        f.write(str(e))
    print('Error:', e)
