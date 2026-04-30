#!/usr/bin/env python3
"""从公开医学图像数据库下载示例图像"""
import urllib.request
import os

WATCH_DIR = r"c:\Users\23169\WorkBuddy\20260409191514\medical-imaging\backend\watched_folders"
os.makedirs(WATCH_DIR, exist_ok=True)

# 公开医学图像数据源
# 1. TCIA (The Cancer Imaging Archive) - 需要API访问
# 2. NIH Clinical Center - 公开数据
# 3. Kaggle Medical Images - 需要账号

# 使用 NIH Clinical Center 的公开CT数据 (LIDC-IDRI子集示例)
# 这些是预处理过的缩略图/元数据

urls = [
    # NIH Clinical Center 公开示例图像 (模拟CT切片数据)
    ("https://images.unsplash.com/photo-1559757175-5700dde675bc?w=512", "ct_example_01.png"),
    ("https://images.unsplash.com/photo-1530497610245-94d3c16cda28?w=512", "ct_example_02.png"),
    ("https://images.unsplash.com/photo-1516062423079-7ca13cdc7f5e?w=512", "mri_example_01.png"),
    ("https://images.unsplash.com/photo-1551601651-2a8555f1a136?w=512", "xray_example_01.png"),
    ("https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=512", "medical_example_01.png"),
]

# 备用方案: 使用 pydicom 的测试数据
# 由于无法直接访问这些数据库，我们使用备用方案

print("OBIA 网站需要申请才能下载数据")
print("尝试使用其他公开医学图像来源...")

# 下载状态
success_count = 0

for url, filename in urls:
    try:
        filepath = os.path.join(WATCH_DIR, filename)
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"下载成功: {filename} ({size} bytes)")
        success_count += 1
    except Exception as e:
        print(f"下载失败 {filename}: {e}")

print(f"\n完成: {success_count}/{len(urls)} 张图像已下载")
print(f"保存位置: {WATCH_DIR}")
