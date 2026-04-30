# -*- coding: utf-8 -*-
import urllib.request, json

req = urllib.request.Request(
    'http://localhost:8000/api/demo-analyze',
    method='POST',
    headers={'Content-Length': '0'}
)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read().decode('utf-8'))

ai = data.get('ai_result', {})
det = data.get('detection', {})

print('ai_result 所有键:', list(ai.keys()))
print('detection 所有键:', list(det.keys()))
print()
print('overall_assessment:', ai.get('overall_assessment', 'N/A')[:200] if ai.get('overall_assessment') else 'N/A')
print('clinical_recommendation:', ai.get('clinical_recommendation', 'N/A'))
print()
print('det.clinical_recommendation:', det.get('clinical_recommendation', 'N/A'))
print('det.overall_assessment:', det.get('overall_assessment', 'N/A')[:200] if det.get('overall_assessment') else 'N/A')
