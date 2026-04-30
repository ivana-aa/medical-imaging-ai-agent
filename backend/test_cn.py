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

print('=== AI分析结论（全中文）===')
print()
print('综合评估:')
print(det.get('overall_assessment', 'N/A'))
print()
print('临床建议:')
print(det.get('clinical_recommendation', 'N/A'))
print()
print('正常发现:')
for f in det.get('normal_summary', [])[:3]:
    print('  - ' + f)
print()
print('异常发现:')
for f in det.get('abnormal_findings', [])[:2]:
    print('  类型: ' + str(f.get('type', 'N/A')))
    print('  位置: ' + str(f.get('location', 'N/A')))
    print('  描述: ' + str(f.get('description', 'N/A')))
print()
print('风险级别: ' + det.get('risk_level', 'N/A'))
print('置信度: ' + str(det.get('confidence', 'N/A')))
print()
print('引擎: ' + data.get('engine', 'N/A'))
