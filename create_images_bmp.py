#!/usr/bin/env python3
"""创建BMP格式医学测试图像"""
import os
import struct
import random

watch_dir = r"c:\Users\23169\WorkBuddy\20260409191514\medical-imaging\backend\watched_folders"
os.makedirs(watch_dir, exist_ok=True)

def create_bmp_gray(name, width=512, height=512):
    """创建8位灰度BMP文件"""
    filepath = os.path.join(watch_dir, name)
    
    # BMP文件头 (14字节)
    file_size = 14 + 40 + 1024 + width * height  # header + DIB + palette + data
    file_header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
    
    # DIB头 (40字节 BITMAPINFOHEADER)
    dib_header = struct.pack('<IiiHHIIiiII', 
        40, width, height * 2,  # height * 2 for top-down
        1, 8, 0, width * height, 0, 0, 0, 0)  # 8-bit gray
    
    # 调色板 (1024字节 = 256 * 4)
    palette = bytes(i for i in range(256) for _ in range(4))
    
    # 像素数据
    pixels = bytearray(width * height)
    for i in range(len(pixels)):
        pixels[i] = random.randint(30, 200)
    
    # 添加圆形结构
    for _ in range(3):
        r = random.randint(50, 150)
        ox, oy = random.randint(100, 400), random.randint(100, 400)
        for y in range(height):
            for x in range(width):
                if (x - ox)**2 + (y - oy)**2 <= r**2:
                    pixels[y * width + x] = random.randint(80, 220)
    
    with open(filepath, 'wb') as f:
        f.write(file_header + dib_header + palette + bytes(pixels))
    
    return filepath

images = [
    ("chest_xray_01.bmp", "Chest X-Ray"),
    ("chest_xray_02.bmp", "Chest X-Ray"),
    ("chest_xray_03.bmp", "Chest X-Ray"),
    ("ct_brain_01.bmp", "CT Brain"),
    ("ct_brain_02.bmp", "CT Brain"),
    ("ct_abdomen_01.bmp", "CT Abdomen"),
    ("ct_abdomen_02.bmp", "CT Abdomen"),
    ("ct_thorax_01.bmp", "CT Thorax"),
    ("ct_thorax_02.bmp", "CT Thorax"),
    ("mri_brain_01.bmp", "MRI Brain"),
    ("mri_brain_02.bmp", "MRI Brain"),
    ("mri_spine_01.bmp", "MRI Spine"),
    ("mri_knee_01.bmp", "MRI Knee"),
    ("xray_spine_01.bmp", "Spine X-Ray"),
    ("xray_hand_01.bmp", "Hand X-Ray"),
    ("xray_skull_01.bmp", "Skull X-Ray"),
    ("dental_xray_01.bmp", "Dental X-Ray"),
    ("ultrasound_01.bmp", "Ultrasound"),
    ("mammogram_01.bmp", "Mammogram"),
    ("ct_liver_01.bmp", "CT Liver"),
]

print("创建合成医学测试图像...")
print(f"目标目录: {watch_dir}")
print()

for name, desc in images:
    create_bmp_gray(name, 512, 512)
    print(f"创建: {name} ({desc}) OK")

print()
print(f"完成! 共 {len(images)} 张图像")
print()

files = os.listdir(watch_dir)
print(f"监控目录 ({len(files)} 个文件):")
for f in sorted(files):
    size = os.path.getsize(os.path.join(watch_dir, f))
    print(f"  {f} ({size//1024} KB)")
