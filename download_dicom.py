#!/usr/bin/env python3
"""下载医学影像示例"""

import os
import urllib.request
import ssl
from pathlib import Path

# SSL设置
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 目标目录
WATCH_DIR = Path(r"c:\Users\23169\WorkBuddy\20260409191514\medical-imaging\backend\watched_folders")
WATCH_DIR.mkdir(parents=True, exist_ok=True)

# DICOM 测试文件URL (来自pydicom官方仓库)
dicom_urls = [
    ("CT_chest_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/CT_small.dcm"),
    ("MR_brain_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/MR_small.dcm"),
    ("CT_scan_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/CT_CHEST.dcm"),
    ("MRbrain_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/MRbrain.dcm"),
    ("CT_thorax_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/rtoberben.dcm"),
    ("CT_abdo_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/CT_small_inv.dcm"),
    ("CT_spine_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/CT_small_pz.dcm"),
    ("CT_jpeg_001.dcm", "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/CT_wJPEG.dcm"),
]

def download_file(url, filepath):
    """下载单个文件"""
    try:
        print(f"下载: {filepath.name}...", end=" ", flush=True)
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=30, context=ssl_context) as resp:
            filepath.write_bytes(resp.read())
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False

print("=" * 50)
print("下载 DICOM 医学影像...")
print("=" * 50)
print(f"保存到: {WATCH_DIR}")
print()

downloaded = 0
for name, url in dicom_urls:
    if download_file(url, WATCH_DIR / name):
        downloaded += 1

# 如果下载不够，用合成图像补充
if downloaded < 20:
    print("\n创建合成测试图像...")
    try:
        from PIL import Image
        import numpy as np
        
        synth_images = [
            "synthetic_xray_chest.png",
            "synthetic_ct_abdomen.png",
            "synthetic_mri_brain.png",
            "synthetic_xray_spine.png",
            "synthetic_ct_thorax.png",
            "synthetic_mri_knee.png",
            "synthetic_xray_hand.png",
            "synthetic_ct_spine.png",
            "synthetic_mri_spine.png",
            "synthetic_xray_skull.png",
            "synthetic_ct_liver.png",
            "synthetic_mri_head.png",
        ]
        
        for name in synth_images:
            filepath = WATCH_DIR / name
            # 创建512x512灰度图像
            img = np.random.randint(40, 180, (512, 512), dtype=np.uint8)
            Image.fromarray(img, mode='L').save(filepath)
            print(f"创建: {name} ✓")
            downloaded += 1
            if downloaded >= 20:
                break
    except ImportError:
        print("PIL未安装，跳过合成图像")

print()
print("=" * 50)
print(f"完成! 共 {downloaded} 张图像")
print("=" * 50)

# 列出文件
files = list(WATCH_DIR.iterdir())
print(f"\n监控目录 ({len(files)} 个文件):")
for f in sorted(files)[:20]:
    size = f.stat().st_size
    print(f"  {f.name} ({size//1024} KB)")
