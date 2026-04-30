# -*- coding: utf-8 -*-
import urllib.request, json, os

base = 'C:/Users/23169/WorkBuddy/20260409191514/medical-imaging'
os.chdir(base)

req = urllib.request.Request(
    'http://localhost:8001/api/demo-analyze',
    method='POST',
    headers={'Content-Length': '0'}
)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read().decode('utf-8'))

det = data.get('detection', {})
ai = data.get('ai_result', {})

with open('backend/demo_result.json', 'w', encoding='utf-8') as f:
    json.dump({
        'engine': data.get('engine'),
        'ai_keys': list(ai.keys()),
        'det_keys': list(det.keys()),
        'clinical': det.get('clinical_recommendation', 'N/A'),
        'overall': str(det.get('overall_assessment', ''))[:150],
        'normal': det.get('normal_summary', [])[:2]
    }, f, ensure_ascii=False, indent=2)
print('Done')
