#!/usr/bin/env python3
"""
下载公开医学影像用于测试
使用多个公开数据源的示例图像
"""

import os
import urllib.request
import ssl
import time

# 创建SSL上下文以避免证书问题
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 目标目录
WATCH_DIR = r"c:\Users\23169\WorkBuddy\20260409191514\medical-imaging\backend\watched_folders"
os.makedirs(WATCH_DIR, exist_ok=True)

def download_file(url, filepath, timeout=60):
    """下载文件"""
    try:
        print(f"下载: {url}")
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        print(f"  ✓ 保存至: {filepath}")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

# 公开医学图像资源
# 这些是示例URL，需要根据实际可用资源调整

images_to_download = []

# 1. NIH Clinical Center 公开示例图像 (DICOM)
# 使用临床影像的公开示例
sample_urls = [
    # DICOM 示例图像 (来自公开医学影像库)
    {
        "name": "chest_xray_001.dcm",
        "url": "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/CT_small.dcm",
        "desc": "CT图像示例"
    },
    {
        "name": "chest_xray_002.dcm", 
        "url": "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/MR_small.dcm",
        "desc": "MR图像示例"
    },
    # 更多公开DICOM测试文件
    {
        "name": "brain_mri_001.dcm",
        "url": "https://raw.githubusercontent.com/pydicom/pydicom/master/data/test_files/CT_scan_J2Ki.dcm",
        "desc": "脑部CT"
    },
]

# 2. 公开医学图像数据集 (PNG/JPEG格式)
# 这些是公开可用的医学影像
medical_images = [
    # Kaggle 医学图像 (示例URL - 需要替换为实际可用资源)
    ("xray_chest_001.png", "https://storage.googleapis.com/kaggle-datasets-images/1844241/3055609/fc6af8be03bd9d7e3c2b9e3c2a4b7e8c/Image-0001.jpeg", "胸部X光"),
    ("xray_chest_002.png", "https://storage.googleapis.com/kaggle-datasets-images/1844241/3055609/fc6af8be03bd9d7e3c2b9e3c2a4b7e8c/Image-0002.jpeg", "胸部X光"),
]

def download_pydicom_samples():
    """下载 pydicom 测试数据集中的 DICOM 文件"""
    print("\n" + "="*50)
    print("下载 DICOM 测试图像...")
    print("="*50)
    
    base_urls = [
        # pydicom 官方测试文件
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/CT_small.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/MR_small.dcm",
        "https://github.com/pydicom/pydicom/raw/refs/heads/master/data/test_files/CT_scan_J2Ki.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/CT_CHEST.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/MRbrain.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/rtoberben.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_filesMR_Spectroscopy_ACR.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/CT_small_inv.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/CT_small_pz.dcm",
        "https://github.com/pydicom/pydicom/raw/master/data/test_files/CT_wJPEG.dcm",
    ]
    
    names = [
        "CT_chest_001.dcm",
        "MR_brain_001.dcm",
        "CT_scan_001.dcm",
        "CT_chest_002.dcm",
        "MRbrain_001.dcm",
        "CT_thorax_001.dcm",
        "CT_abdo_001.dcm",
        "CT_chest_003.dcm",
        "CT_spine_001.dcm",
        "CT_jpeg_001.dcm",
    ]
    
    success_count = 0
    for i, url in enumerate(base_urls):
        # 替换 raw.githubusercontent.com 为 raw.githubusercontent.com
        url = url.replace("refs/heads/", "")
        filepath = os.path.join(WATCH_DIR, names[i] if i < len(names) else f"dicom_{i:03d}.dcm")
        if download_file(url, filepath):
            success_count += 1
        time.sleep(0.5)  # 避免请求过快
    
    return success_count

def create_synthetic_images():
    """创建合成的医学图像用于测试（作为备用）"""
    print("\n" + "="*50)
    print("创建合成测试图像...")
    print("="*50)
    
    try:
        from PIL import Image
        import numpy as np
        
        # 创建不同类型的合成医学图像
        synthetic_images = [
            ("synthetic_chest_xray.png", "Chest X-ray", (512, 512)),
            ("synthetic_ct_scan.png", "CT Scan Slice", (256, 256)),
            ("synthetic_mri_brain.png", "MRI Brain", (400, 400)),
            ("synthetic_xray_spine.png", "Spine X-ray", (300, 400)),
            ("synthetic_ct_abdomen.png", "CT Abdomen", (350, 350)),
            ("synthetic_mri_knee.png", "MRI Knee", (320, 320)),
            ("synthetic_xray_hand.png", "Hand X-ray", (280, 400)),
            ("synthetic_ct_thorax.png", "CT Thorax", (360, 360)),
            ("synthetic_mri_spine.png", "MRI Spine", (300, 450)),
            ("synthetic_xray_skull.png", "Skull X-ray", (350, 380)),
        ]
        
        for name, title, size in synthetic_images:
            filepath = os.path.join(WATCH_DIR, name)
            
            # 创建灰度图像模拟医学影像
            img_array = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
            
            # 添加一些圆形/椭圆形模拟器官或异常区域
            center_x, center_y = size[0]//2, size[1]//2
            
            # 创建更真实的医学图像效果
            x = np.linspace(0, size[0]-1, size[0])
            y = np.linspace(0, size[1]-1, size[1])
            X, Y = np.meshgrid(x, y)
            
            # 模拟圆形结构（类似器官或组织）
            Z = np.zeros((*size[::-1], 3), dtype=np.uint8)
            for _ in range(5):
                cx = np.random.randint(50, size[0]-50)
                cy = np.random.randint(50, size[1]-50)
                rx = np.random.randint(20, 80)
                ry = np.random.randint(20, 80)
                ellipse_mask = ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1
                Z[ellipse_mask] = np.random.randint(100, 200, 3)
            
            # 添加噪点
            noise = np.random.randint(-20, 20, (*size[::-1], 3), dtype=np.int16)
            Z = np.clip(Z.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(Z)
            img.save(filepath)
            print(f"  ✓ 创建: {filepath} ({title})")
        
        return len(synthetic_images)
    except ImportError:
        print("  ! PIL 未安装，跳过合成图像创建")
        return 0

def main():
    print("="*60)
    print("医学影像下载工具")
    print("="*60)
    print(f"目标目录: {WATCH_DIR}")
    print()
    
    total = 0
    
    # 1. 尝试下载 DICOM 测试文件
    dcm_count = download_pydicom_samples()
    total += dcm_count
    
    # 2. 如果下载的DICOM不够20张，创建合成图像
    if total < 20:
        synth_count = create_synthetic_images()
        total += synth_count
    
    print("\n" + "="*60)
    print(f"下载完成! 共 {total} 张图像")
    print(f"保存位置: {WATCH_DIR}")
    print("="*60)
    
    # 列出下载的文件
    files = os.listdir(WATCH_DIR)
    print(f"\n目录中共有 {len(files)} 个文件:")
    for f in sorted(files)[:25]:  # 最多显示25个
        size = os.path.getsize(os.path.join(WATCH_DIR, f))
        print(f"  - {f} ({size/1024:.1f} KB)")
    if len(files) > 25:
        print(f"  ... 还有 {len(files)-25} 个文件")

if __name__ == "__main__":
    main()
