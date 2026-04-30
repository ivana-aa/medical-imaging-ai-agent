#!/usr/bin/env python3
"""快速创建合成医学测试图像"""
import os
import sys

try:
    from PIL import Image
except ImportError:
    print("需要安装 Pillow: pip install Pillow")
    sys.exit(1)

watch_dir = r"c:\Users\23169\WorkBuddy\20260409191514\medical-imaging\backend\watched_folders"
os.makedirs(watch_dir, exist_ok=True)

images = [
    ("chest_xray_01.png", "Chest X-Ray"),
    ("chest_xray_02.png", "Chest X-Ray"),
    ("chest_xray_03.png", "Chest X-Ray"),
    ("ct_brain_01.png", "CT Brain"),
    ("ct_brain_02.png", "CT Brain"),
    ("ct_abdomen_01.png", "CT Abdomen"),
    ("ct_abdomen_02.png", "CT Abdomen"),
    ("ct_thorax_01.png", "CT Thorax"),
    ("ct_thorax_02.png", "CT Thorax"),
    ("mri_brain_01.png", "MRI Brain"),
    ("mri_brain_02.png", "MRI Brain"),
    ("mri_spine_01.png", "MRI Spine"),
    ("mri_knee_01.png", "MRI Knee"),
    ("xray_spine_01.png", "Spine X-Ray"),
    ("xray_hand_01.png", "Hand X-Ray"),
    ("xray_skull_01.png", "Skull X-Ray"),
    ("dental_xray_01.png", "Dental X-Ray"),
    ("ultrasound_01.png", "Ultrasound"),
    ("mammogram_01.png", "Mammogram"),
    ("ct_liver_01.png", "CT Liver"),
]

import numpy as np

for name, _ in images:
    img = np.random.randint(30, 200, (512, 512), dtype=np.uint8)
    for _ in range(3):
        cx, cy = np.random.randint(100, 400, 2)
        r = np.random.randint(30, 100)
        y, x = np.ogrid[:512, :512]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = np.random.randint(80, 220)
    filepath = os.path.join(watch_dir, name)
    Image.fromarray(img, mode="L").save(filepath)
    print(f"创建: {name}")

print(f"\n完成! 共 {len(images)} 张图像")
print(f"保存位置: {watch_dir}")
