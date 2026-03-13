#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机图像拼接脚本 - 简单版本
功能：设置裁剪参数，找到对齐的两帧，生成拼接图像
"""

import rosbag
import numpy as np
import cv2
from pathlib import Path


class CameraStitcher:
    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)

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

    def _decode_image(self, ros_image_data, width, height, encoding):
        """解码ROS Image消息"""
        if encoding == 'rgb8':
            img = np.frombuffer(ros_image_data, dtype=np.uint8).reshape(height, width, 3)
        elif encoding == 'bgr8':
            img = np.frombuffer(ros_image_data, dtype=np.uint8).reshape(height, width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif encoding == 'mono8':
            img = np.frombuffer(ros_image_data, dtype=np.uint8).reshape(height, width)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = np.frombuffer(ros_image_data, dtype=np.uint8).reshape(height, width, 3)
        return img

    def _load_images(self):
        """从Bag文件加载图像"""
        print(f"加载 {self.bag_path}...")
        with rosbag.Bag(str(self.bag_path), 'r') as bag:
            for topic, msg, t in bag.read_messages(
                topics=['/camera/left_front_view', '/camera/right_front_view']
            ):
                timestamp = t.to_sec()
                img = self._decode_image(msg.data, msg.width, msg.height, msg.encoding)

                if topic == '/camera/left_front_view':
                    self.left_images[timestamp] = img
                else:
                    self.right_images[timestamp] = img

        print(f"左图：{len(self.left_images)} 帧，右图：{len(self.right_images)} 帧")

    def stitch_frame(self, left_idx=0, right_idx=None):
        """
        拼接指定帧的左右图像

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
        print(f"拼接结果：{stitched.shape}")

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
    bag_path = '/working-space/wky/bags/sensor_data_1.bag'

    stitcher = CameraStitcher(bag_path)

    # 列出可用帧
    stitcher.list_frames()

    # === 修改这里设置裁剪参数 ===
    # 左摄像头
    stitcher.left_crop_h_start = 0          # ← 上方裁剪 (0~1080)
    stitcher.left_crop_h_end = 1080-48-172      # ← 下方保留 (0~1080)
    stitcher.left_crop_w_start = 0        # ← 左边裁剪 (0~1920)
    stitcher.left_crop_w_end = 1920-200     # ← 右边保留 (0~1920)

    # 右摄像头（可与左不同）
    stitcher.right_crop_h_start = 48        # ← 上方裁剪 (0~1080)
    stitcher.right_crop_h_end = 1080-172        # ← 下方保留 (0~1080)
    stitcher.right_crop_w_start = 200       # ← 左边裁剪 (0~1920)
    stitcher.right_crop_w_end = 1920    # ← 右边保留 (0~1920)

    # 拼接第一对对齐的图像
    stitched = stitcher.stitch_frame(left_idx=0, right_idx=0)

    if stitched is not None:
        stitcher.save(stitched, 'stitched_result.png')


if __name__ == '__main__':
    main()
