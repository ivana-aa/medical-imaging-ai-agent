# -*- coding: utf-8 -*-
import urllib.request, json

req = urllib.request.Request(
    'http://localhost:8000/api/demo-analyze',
    method='POST',
    headers={'Content-Length': '0'}
)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read().decode('utf-8'))

# 直接看 ai_result 原始数据
ai = data.get('ai_result', {})
print('=== ai_result 原始数据 ===')
print('所有键:', list(ai.keys()))
print()
print('overall_assessment:', ai.get('overall_assessment', 'N/A'))
print('clinical_recommendation:', ai.get('clinical_recommendation', 'N/A'))
print()
print('=== detection 字段 ===')
det = data.get('detection', {})
print('所有键:', list(det.keys()))
