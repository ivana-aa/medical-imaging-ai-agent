# -*- coding: utf-8 -*-
import sys, os, json
sys.path.insert(0, '.')
os.chdir('C:/Users/23169/WorkBuddy/20260409191514/medical-imaging')

import numpy as np
from backend.main import analyzer

np.random.seed(42)
img_array = np.zeros((512, 512), dtype=np.uint8)
for i in range(512):
    for j in range(512):
        img_array[i, j] = int(128 + 50 * np.sin(i/30) * np.cos(j/30) + 20 * np.random.random())

b64 = analyzer.image_to_base64(img_array)
result = analyzer.call_vision_api(b64, 'test.jpg', 'chest')

with open('backend/test_result2.json', 'w', encoding='utf-8') as f:
    json.dump({
        'result_type': type(result).__name__,
        'result_keys': list(result.keys()) if isinstance(result, dict) else None,
        'clinical_recommendation': result.get('clinical_recommendation') if isinstance(result, dict) else None,
        'has_result': result is not None
    }, f, ensure_ascii=False, indent=2)
print('Done')
