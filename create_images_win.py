#!/usr/bin/env python3
"""使用纯Python创建PPM格式图像（不需要Pillow）"""
import os
import struct
import random

watch_dir = r"c:\Users\23169\WorkBuddy\20260409191514\medical-imaging\backend\watched_folders"
os.makedirs(watch_dir, exist_ok=True)

def create_ppm_gray(width, height, name):
    """创建PGM格式灰度图像"""
    filepath = os.path.join(watch_dir, name)
    
    with open(filepath, 'wb') as f:
        # PGM header
        f.write(f"P5\n{width} {height}\n255\n".encode())
        
        # 图像数据
        pixels = bytearray(width * height)
        for i in range(len(pixels)):
            pixels[i] = random.randint(30, 200)
        
        # 添加圆形结构模拟器官
        cx, cy = width // 2, height // 2
        r = min(width, height) // 4
        for y in range(height):
            for x in range(width):
                if (x - cx)**2 + (y - cy)**2 <= r**2:
                    pixels[y * width + x] = random.randint(100, 220)
        
        f.write(pixels)
    
    return filepath

def create_png_header(name, width=512, height=512):
    """创建最小PNG文件（固定内容作为测试）"""
    filepath = os.path.join(watch_dir, name)
    
    # PNG 文件签名
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr_crc = zlib_crc(b'IHDR' + ihdr_data)
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + ihdr_crc
    
    # IDAT chunk (简单的压缩数据)
    raw_data = bytes([random.randint(20, 180) for _ in range(width * height * 3)])
    compressed = zlib_compress(raw_data)
    idat_crc = zlib_crc(b'IDAT' + compressed)
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + idat_crc
    
    # IEND chunk
    iend_crc = zlib_crc(b'IEND')
    iend = struct.pack('>I') + b'IEND' + iend_crc
    
    with open(filepath, 'wb') as f:
        f.write(signature + ihdr + idat + iend)
    
    return filepath

def zlib_compress(data):
    """简单的zlib压缩（使用Python内置）"""
    import zlib
    return zlib.compress(data)

def zlib_crc(data):
    """计算CRC32"""
    import zlib
    return struct.pack('>I', zlib.crc32(data) & 0xffffffff)

images = [
    "chest_xray_01.png",
    "chest_xray_02.png",
    "chest_xray_03.png",
    "ct_brain_01.png",
    "ct_brain_02.png",
    "ct_abdomen_01.png",
    "ct_abdomen_02.png",
    "ct_thorax_01.png",
    "ct_thorax_02.png",
    "mri_brain_01.png",
    "mri_brain_02.png",
    "mri_spine_01.png",
    "mri_knee_01.png",
    "xray_spine_01.png",
    "xray_hand_01.png",
    "xray_skull_01.png",
    "dental_xray_01.png",
    "ultrasound_01.png",
    "mammogram_01.png",
    "ct_liver_01.png",
]

print("创建合成医学测试图像...")
print(f"目标目录: {watch_dir}")
print()

for name in images:
    try:
        create_png_header(name, 512, 512)
        print(f"创建: {name} ✓")
    except Exception as e:
        print(f"创建 {name} 失败: {e}")

print()
print(f"完成! 共 {len(images)} 张图像")

# 列出文件
files = os.listdir(watch_dir)
print(f"\n监控目录 ({len(files)} 个文件):")
for f in sorted(files):
    size = os.path.getsize(os.path.join(watch_dir, f))
    print(f"  {f} ({size//1024} KB)")
