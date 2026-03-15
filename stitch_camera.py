#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机图像拼接脚本 - 快速版本 (带本地缓存)
功能：首次从 rosbag 提取并缓存图像对，之后直接使用本地缓存，快速生成裁剪拼接图
"""

import rosbag
import numpy as np
import cv2
from pathlib import Path
import hashlib
import time


class CameraStitcher:
    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)

        # 缓存目录：当前工作目录 + cache 子目录
        self.cache_dir = Path.cwd() / "cache" / self._get_bag_hash()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存文件路径
        self.left_cache_path = self.cache_dir / "left_image.png"
        self.right_cache_path = self.cache_dir / "right_image.png"
        self.raw_stitch_cache_path = self.cache_dir / "raw_stitched.png"
        self.metadata_path = self.cache_dir / "metadata.txt"

        # 左摄像头：上下裁剪参数
        self.left_crop_h_start = 60
        self.left_crop_h_end = 1020
        # 左摄像头：左右裁剪参数
        self.left_crop_w_start = 0
        self.left_crop_w_end = 1920

        # 右摄像头：上下裁剪参数
        self.right_crop_h_start = 60
        self.right_crop_h_end = 1020
        # 右摄像头：左右裁剪参数
        self.right_crop_w_start = 0
        self.right_crop_w_end = 1920

        # 加载图像
        self.left_images = {}  # {timestamp: image}
        self.right_images = {}  # {timestamp: image}
        self._load_images()

    def _get_bag_hash(self):
        """获取 bag 文件的哈希值作为缓存标识"""
        with open(self.bag_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]

    def _decode_compressed_image(self, compressed_data):
        """解码 CompressedImage 压缩数据"""
        nparr = np.frombuffer(compressed_data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 读取为 BGR
        if img is None:
            raise ValueError("无法解码压缩图像数据")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        return img

    def _load_images_from_bag(self):
        """从 Bag 文件加载压缩图像，找到第一对匹配的左右图像后停止"""
        print(f"📦 从 rosbag 读取图像：{self.bag_path.name}...")
        t0 = time.time()

        last_left_time = None
        last_right_time = None
        time_match_threshold = 0.1  # 100ms 内认为匹配

        with rosbag.Bag(str(self.bag_path), 'r') as bag:
            for topic, msg, t in bag.read_messages(
                topics=['/camera/left_front_view/compressed', '/camera/right_front_view/compressed']
            ):
                timestamp = t.to_sec()
                try:
                    img = self._decode_compressed_image(msg.data)
                except Exception as e:
                    print(f"⚠️  解码失败 {topic} @ {timestamp}: {e}")
                    continue

                if topic == '/camera/left_front_view/compressed':
                    last_left_time = timestamp
                    self.left_images[timestamp] = img

                    # 检查是否有匹配的右图
                    if last_right_time is not None and abs(last_left_time - last_right_time) < time_match_threshold:
                        print(f"✓ 找到第一对匹配图像 (时间差: {abs(last_left_time - last_right_time):.6f}s)")
                        break

                else:  # 右图
                    last_right_time = timestamp
                    self.right_images[timestamp] = img

                    # 检查是否有匹配的左图
                    if last_left_time is not None and abs(last_left_time - last_right_time) < time_match_threshold:
                        print(f"✓ 找到第一对匹配图像 (时间差: {abs(last_left_time - last_right_time):.6f}s)")
                        break

        elapsed = time.time() - t0
        print(f"✓ 读取完成 (耗时 {elapsed:.2f}s)：左图 {len(self.left_images)} 帧，右图 {len(self.right_images)} 帧")

    def _save_cache(self):
        """保存第一对匹配的左右图像到缓存"""
        if not self.left_images or not self.right_images:
            print("❌ 没有可用的图像数据")
            return False

        # 找出时间戳最接近的左右图像对
        left_times = sorted(self.left_images.keys())
        right_times = sorted(self.right_images.keys())

        # 查找第一对匹配的图像（时间戳最接近）
        best_left_idx, best_right_idx = 0, 0
        min_time_diff = float('inf')

        for i, left_t in enumerate(left_times):
            for j, right_t in enumerate(right_times):
                time_diff = abs(left_t - right_t)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_left_idx = i
                    best_right_idx = j
                # 如果找到时间戳完全相同的，就用这一对
                if time_diff == 0:
                    break
            if min_time_diff == 0:
                break

        left_time = left_times[best_left_idx]
        right_time = right_times[best_right_idx]
        left_img = self.left_images[left_time]
        right_img = self.right_images[right_time]

        print(f"✓ 选中匹配对：左 #{best_left_idx} (t={left_time:.6f}) <-> 右 #{best_right_idx} (t={right_time:.6f})")
        print(f"  时间差：{min_time_diff:.6f}s")

        # 保存单张图像
        cv2.imwrite(str(self.left_cache_path), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.right_cache_path), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))

        # 生成原始拼接图
        if left_img.shape[0] != right_img.shape[0]:
            min_h = min(left_img.shape[0], right_img.shape[0])
            left_img = left_img[:min_h]
            right_img = right_img[:min_h]

        raw_stitch = np.concatenate([left_img, right_img], axis=1)
        cv2.imwrite(str(self.raw_stitch_cache_path), cv2.cvtColor(raw_stitch, cv2.COLOR_RGB2BGR))

        # 保存元数据
        with open(self.metadata_path, 'w') as f:
            f.write(f"left_time: {left_time:.6f}\n")
            f.write(f"right_time: {right_time:.6f}\n")
            f.write(f"time_diff: {min_time_diff:.6f}\n")
            f.write(f"raw_stitch_shape: {raw_stitch.shape}\n")

        print(f"💾 缓存已保存到：{self.cache_dir}")
        print(f"   - {self.left_cache_path.name} ({left_img.shape})")
        print(f"   - {self.right_cache_path.name} ({right_img.shape})")
        print(f"   - {self.raw_stitch_cache_path.name} ({raw_stitch.shape})")
        return True

    def _load_from_cache(self):
        """从本地缓存加载图像"""
        if not self.left_cache_path.exists() or not self.right_cache_path.exists():
            return False

        print(f"⚡ 从本地缓存加载图像...")
        try:
            # 读取缓存的左右图
            left_bgr = cv2.imread(str(self.left_cache_path))
            right_bgr = cv2.imread(str(self.right_cache_path))

            if left_bgr is None or right_bgr is None:
                return False

            left_img = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

            # 仅保存一对图像（时间戳为 0）
            self.left_images = {0: left_img}
            self.right_images = {0: right_img}

            print(f"✓ 缓存加载成功：{left_img.shape}, {right_img.shape}")

            # 检查原始拼接图
            if self.raw_stitch_cache_path.exists():
                print(f"✓ 原始拼接图已存在：{self.raw_stitch_cache_path.name}")

            return True
        except Exception as e:
            print(f"⚠️  缓存加载失败：{e}")
            return False

    def _load_images(self):
        """加载或创建缓存的图像"""
        # 先尝试从本地缓存加载
        if self._load_from_cache():
            return

        # 缓存不存在，从 rosbag 读取
        self._load_images_from_bag()

        # 保存缓存供后续使用
        self._save_cache()

    def stitch_frame(self, left_idx=0, right_idx=None):
        """
        拼接指定帧的左右图像（应用裁剪参数）

        Args:
            left_idx: 左图在排序后列表中的索引
            right_idx: 右图在排序后列表中的索引，默认=left_idx
        """
        left_times = sorted(self.left_images.keys())
        right_times = sorted(self.right_images.keys())

        if right_idx is None:
            right_idx = left_idx

        if left_idx >= len(left_times) or right_idx >= len(right_times):
            print(f"❌ 索引越界 (左:{left_idx}/{len(left_times)}, 右:{right_idx}/{len(right_times)})")
            return None

        left_time = left_times[left_idx]
        right_time = right_times[right_idx]
        left_img = self.left_images[left_time]
        right_img = self.right_images[right_time]

        print(f"\n【拼接信息】")
        print(f"左图：#{left_idx} (t={left_time:.3f})")
        print(f"右图：#{right_idx} (t={right_time:.3f})")
        print(f"时间差：{abs(left_time - right_time):.6f}s")
        print(f"左图上下裁剪：{self.left_crop_h_start} ~ {self.left_crop_h_end}")
        print(f"左图左右裁剪：{self.left_crop_w_start} ~ {self.left_crop_w_end}")
        print(f"右图上下裁剪：{self.right_crop_h_start} ~ {self.right_crop_h_end}")
        print(f"右图左右裁剪：{self.right_crop_w_start} ~ {self.right_crop_w_end}")

        # 分别对左右图应用各自的裁剪参数
        left_crop = left_img[
            self.left_crop_h_start:self.left_crop_h_end,
            self.left_crop_w_start:self.left_crop_w_end
        ]
        right_crop = right_img[
            self.right_crop_h_start:self.right_crop_h_end,
            self.right_crop_w_start:self.right_crop_w_end
        ]

        # 如果两张图高度不同，需要调整到相同高度再拼接
        if left_crop.shape[0] != right_crop.shape[0]:
            min_h = min(left_crop.shape[0], right_crop.shape[0])
            left_crop = left_crop[:min_h]
            right_crop = right_crop[:min_h]
            print(f"⚠️  高度不一致，已调整为：{min_h}")

        stitched = np.concatenate([left_crop, right_crop], axis=1)
        print(f"【裁剪后拼接结果】：{stitched.shape}")

        return stitched

    def stitch_frame_raw(self, left_idx=0, right_idx=None):
        """
        拼接指定帧的左右原始图像（不应用裁剪）

        Args:
            left_idx: 左图在排序后列表中的索引
            right_idx: 右图在排序后列表中的索引，默认=left_idx
        """
        left_times = sorted(self.left_images.keys())
        right_times = sorted(self.right_images.keys())

        if right_idx is None:
            right_idx = left_idx

        if left_idx >= len(left_times) or right_idx >= len(right_times):
            print(f"❌ 索引越界 (左:{left_idx}/{len(left_times)}, 右:{right_idx}/{len(right_times)})")
            return None

        left_time = left_times[left_idx]
        right_time = right_times[right_idx]
        left_img = self.left_images[left_time]
        right_img = self.right_images[right_time]

        # 使用原始图像（不裁剪）
        # 如果两张图高度不同，调整到相同高度再拼接
        if left_img.shape[0] != right_img.shape[0]:
            min_h = min(left_img.shape[0], right_img.shape[0])
            left_img = left_img[:min_h]
            right_img = right_img[:min_h]
            print(f"⚠️  原始图像高度不一致，已调整为：{min_h}")

        stitched = np.concatenate([left_img, right_img], axis=1)
        print(f"【原始未裁剪拼接结果】：{stitched.shape}")

        return stitched

    def list_frames(self):
        """列出所有可用帧"""
        left_times = sorted(self.left_images.keys())
        right_times = sorted(self.right_images.keys())

        print(f"\n【左摄像头 - {len(left_times)} 帧】")
        for i, t in enumerate(left_times[:5]):
            print(f"  {i}: t={t:.3f}")
        if len(left_times) > 5:
            print(f"  ... 共 {len(left_times)} 帧")

        print(f"\n【右摄像头 - {len(right_times)} 帧】")
        for i, t in enumerate(right_times[:5]):
            print(f"  {i}: t={t:.3f}")
        if len(right_times) > 5:
            print(f"  ... 共 {len(right_times)} 帧")

    def save(self, image, filename='stitched.png'):
        """保存图像"""
        if image is None:
            return
        path = Path(filename)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ 已保存：{path}")


def main():
    bag_path = '/home/classlab/working-space/wky/bags/2025-03-15/20260315_161247_NUC_0.bag'

    stitcher = CameraStitcher(bag_path)

    # 列出可用帧
    stitcher.list_frames()

    # === 修改这里设置裁剪参数 ===

    raw_h = 1080
    raw_w = 1920
    clip_w = 224
    clip_h = 60
    final_w = raw_w - clip_w
    final_h = final_w//2
    assert final_h < raw_h
    # 左摄像头
    stitcher.left_crop_h_start = 0          # ← 上方裁剪 (0~1080)
    stitcher.left_crop_h_end = final_h      # ← 下方保留 (0~1080)
    stitcher.left_crop_w_end = raw_w-clip_w     # ← 右边保留 (0~1920)
    stitcher.left_crop_w_start = stitcher.left_crop_w_end-final_w       # ← 左边裁剪 (0~1920)

    # 右摄像头（可与左不同）
    stitcher.right_crop_h_start = clip_h        # ← 上方裁剪 (0~1080)
    stitcher.right_crop_h_end = final_h + clip_h       # ← 下方保留 (0~1080)
    stitcher.right_crop_w_start = clip_w       # ← 左边裁剪 (0~1920)
    stitcher.right_crop_w_end = final_w+stitcher.right_crop_w_start    # ← 右边保留 (0~1920)

    # 拼接第一对对齐的图像
    # 1. 原始未裁剪的拼接图像（可从缓存直接读取）
    if stitcher.raw_stitch_cache_path.exists():
        print(f"\n⚡ 原始拼接图已在缓存中：{stitcher.raw_stitch_cache_path.name}")
    else:
        stitched_raw = stitcher.stitch_frame_raw(left_idx=0, right_idx=0)
        if stitched_raw is not None:
            stitcher.save(stitched_raw, 'stitched_raw.png')

    # 2. 裁剪后的拼接图像
    stitched = stitcher.stitch_frame(left_idx=0, right_idx=0)
    if stitched is not None:
        stitcher.save(stitched, 'stitched_result.png')


if __name__ == '__main__':
    main()
