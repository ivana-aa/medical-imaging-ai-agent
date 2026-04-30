# -*- coding: utf-8 -*-
import urllib.request, json

req = urllib.request.Request(
    'http://localhost:8001/api/demo-analyze',
    method='POST',
    headers={'Content-Length': '0'}
)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read().decode('utf-8'))

det = data.get('detection', {})
ai = data.get('ai_result', {})

with open('backend/test_8001_result.json', 'w', encoding='utf-8') as f:
    json.dump({
        'ai_keys': list(ai.keys()),
        'det_keys': list(det.keys()),
        'clinical_rec': det.get('clinical_recommendation', 'N/A'),
        'overall': det.get('overall_assessment', 'N/A')[:100] if det.get('overall_assessment') else 'N/A',
        'engine': data.get('engine', 'N/A'),
        'normal_summary': det.get('normal_summary', [])[:3]
    }, f, ensure_ascii=False, indent=2)
print('Done')
