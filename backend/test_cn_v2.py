# -*- coding: utf-8 -*-
import urllib.request, json

req = urllib.request.Request(
    'http://localhost:8000/api/demo-analyze',
    method='POST',
    headers={'Content-Length': '0'}
)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read().decode('utf-8'))

det = data.get('detection', {})
ai = data.get('ai_result', {})

print('ai_result 所有键:', list(ai.keys()))
print()
print('detection 所有键:', list(det.keys()))
print()
print('ai_result.clinical_recommendation:', ai.get('clinical_recommendation', 'N/A'))
print('detection.clinical_recommendation:', det.get('clinical_recommendation', 'N/A'))
