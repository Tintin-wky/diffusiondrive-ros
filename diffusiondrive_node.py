#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiffusionDrive ROS1 Node - 适配 TransFuser 模型 + 多传感器输入
订阅:
- /bynav/inspvax (novatel_oem7_msgs/INSPVAX): 位姿/速度/姿态
- /camera/left_front_view (sensor_msgs/Image): 前视图像
- /rslidar_points (sensor_msgs/PointCloud2): 点云
发布:
- /diffusiondrive/predicted_trajectory (PoseArray): 预测轨迹（世界系）
- /diffusiondrive/predicted_path (Path): 预测路径
无需 catkin 包，直接 python3 运行
"""
# ==============================================================================
# 第一部分：环境配置（必须在其他导入之前）
# ==============================================================================
import torch
import os
import sys
# 禁用 torch.compile
os.environ["TORCH_COMPILE_BACKEND"] = ""
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disabled = True
if hasattr(torch, 'compile'):
    def _no_compile(fn, *args, **kwargs):
        return fn
    torch.compile = _no_compile
print("[INFO] Disabled torch.compile for ROS compatibility")

# 修复 torch.distributed
try:
    import torch.distributed as dist
except ImportError:
    class FakeDist:
        def is_initialized(self): return False
        def get_rank(self): return 0
        def get_world_size(self): return 1
    dist = FakeDist()
    torch.distributed = dist
if not hasattr(dist, 'is_initialized'):
    dist.is_initialized = lambda: False
if not hasattr(dist, 'get_rank'):
    dist.get_rank = lambda: 0
if not hasattr(dist, 'get_world_size'):
    dist.get_world_size = lambda: 1
print("[INFO] Patched torch.distributed")

# ==============================================================================
# 第二部分：标准导入
# ==============================================================================
import rospy
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import cv2
from PIL import Image as PILImage
from torchvision import transforms
import math
import time
import struct

# ROS 消息类型
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, Twist
from novatel_oem7_msgs.msg import INSPVAX  # NovAtel 惯导消息

# TransFuser/DiffusionDrive 导入
from model.transfuser_agent import TransfuserAgent
from model.transfuser_config import TransfuserConfig
from model.transfuser_features import TransfuserFeatureBuilder

# ==============================================================================
# 第三部分：配置类
# ==============================================================================
class DiffusionDriveROSConfig:
    """ROS 节点配置 - 适配 TransFuser 模型"""
    def __init__(self):
        # ========== 模型路径 ==========
        self.checkpoint = '/root/diffusiondrive/ckpts/diffusiondrive_navsim_88p1_PDMS.pth'
        
        # ========== 模型配置 ==========
        self.action_horizon = 8
        self.action_dim = 3  # [x, y, heading]
        
        # ========== ROS Topic 配置 ==========
        self.inspvax_topic = "/bynav/inspvax"
        self.image_left_topic = "/camera/left_front_view"      # 左前视
        self.image_right_topic = "/camera/right_front_view"    # 右前视
        self.camera_left_info_topic = "/camera/left_front_view/camera_info"
        self.camera_right_info_topic = "/camera/right_front_view/camera_info"
        self.pointcloud_topic = "/rslidar_points"
        self.trajectory_pub_topic = "/diffusiondrive/predicted_trajectory"
        self.path_pub_topic = "/diffusiondrive/predicted_path"
        
        # ========== 推理配置 ==========
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history_length = 4
        self.command_one_hot = [0, 1, 0, 0]  # 默认直行
        self.inference_rate = 10  # Hz

        # ========== 坐标系参数 ==========
        self.lat_ref = None
        self.lon_ref = None
        
        # ========== 激光雷达配置 (参考 diffusiondrive_b2d_agent.py) ==========
        # ego → NuScenes LiDAR 转换矩阵 (lidar2ego 的逆)
        self.ego2lidar = np.array([
            [ 0., -1.,  0.,  0.  ],
            [ 1.,  0.,  0.,  0.39],
            [ 0.,  0.,  1., -1.84],
            [ 0.,  0.,  0.,  1.  ],
        ])
        
        # LiDAR → 前视摄像头像素
        self.lidar2img_front = np.array([
            [ 1.14251841e+03,  8.00000000e+02,  0.0,             -9.52000000e+02],
            [ 0.0,             4.50000000e+02, -1.14251841e+03,  -8.09704417e+02],
            [ 0.0,             1.00000000e+00,  0.0,             -1.19000000e+00],
            [ 0.0,             0.0,             0.0,              1.00000000e+00],
        ])
        
        # LiDAR → Ego 转换矩阵 (用于点云坐标转换)
        self.lidar2ego = np.array([
            [ 0.,  1.,  0., -0.39],
            [-1.,  0.,  0.,  0.  ],
            [ 0.,  0.,  1.,  1.84],
            [ 0.,  0.,  0.,  1.  ],
        ])
        
        # ========== BEV 直方图配置 ==========
        self.lidar_min_x = -32.0
        self.lidar_max_x = 32.0
        self.lidar_min_y = -32.0
        self.lidar_max_y = 32.0
        self.pixels_per_meter = 8.0
        self.max_height_lidar = 2.0
        self.lidar_split_height = 0.0
        self.hist_max_per_pixel = 100
        
        # 图像配置 (1920x1080 → 4:1 裁剪)
        self.image_width = 1920
        self.image_height = 1080
        self.crop_h_start = 60
        self.crop_h_end = 1020
        self.camera_feature_width = 1024
        self.camera_feature_height = 256

        # ========== 打印配置 ==========
        print("=" * 60)
        print("DiffusionDrive ROS 配置")
        print("=" * 60)
        print(f"设备：{self.device}")
        print(f"BEV 范围：[{self.lidar_min_x}, {self.lidar_max_x}] x [{self.lidar_min_y}, {self.lidar_max_y}]")
        print(f"BEV 分辨率：{int((self.lidar_max_x - self.lidar_min_x) * self.pixels_per_meter)}x"
              f"{int((self.lidar_max_y - self.lidar_min_y) * self.pixels_per_meter)}")
        print("=" * 60)


# ==============================================================================
# 第四部分：坐标转换工具函数
# ==============================================================================
def gps_to_enu(lat: float, lon: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    """GPS (lat/lon) → 局部 ENU 坐标"""
    EARTH_RADIUS_EQUA = 6378137.0
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
    mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
    y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my
    x = mx - scale * lat_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    return x, y


def azimuth_to_yaw(azimuth_deg: float) -> float:
    """INSPVAX azimuth → 模型 yaw"""
    azimuth_rad = math.radians(azimuth_deg)
    yaw = math.pi / 2 - azimuth_rad
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return yaw


def enu_to_ego_local(dx: float, dy: float, current_yaw: float) -> Tuple[float, float]:
    """ENU 位移 → 自车局部坐标系 (X-fwd, Y-left)"""
    cos_yaw = math.cos(current_yaw)
    sin_yaw = math.sin(current_yaw)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    return local_x, local_y


def convert_history_to_local(global_poses: List[Tuple[float, float, float]]) -> np.ndarray:
    """将全球坐标系历史轨迹转换到模型局部坐标系"""
    if len(global_poses) < 4:
        poses = global_poses + [global_poses[-1]] * (4 - len(global_poses))
    else:
        poses = global_poses[-4:]
    
    x0, y0, yaw0 = poses[-1]
    cos_h0 = math.cos(yaw0)
    sin_h0 = math.sin(yaw0)
    
    local_poses = []
    for x, y, yaw in poses:
        dx = x - x0
        dy = y - y0
        local_x, local_y = enu_to_ego_local(dx, dy, yaw0)
        heading_local = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
        local_poses.append([local_x, local_y, heading_local])
    
    return np.array(local_poses, dtype=np.float32)


# ==============================================================================
# 第五部分：数据缓存类
# ==============================================================================
class DataBuffer:
    """ROS 数据缓存"""
    def __init__(self, history_length: int = 4):
        self.history_length = history_length
        self.global_pose_buffer = deque(maxlen=history_length)
        self.velocity_enu_buffer = deque(maxlen=history_length)
        
        self.latest_inspvax: Optional[INSPVAX] = None
        self.latest_image_left: Optional[Image] = None
        self.latest_image_right: Optional[Image] = None
        self.latest_pointcloud: Optional[PointCloud2] = None
        
        self.lat_ref = None
        self.lon_ref = None
        self.initialized = False
        self.camera_left_info: Optional[CameraInfo] = None
        self.camera_right_info: Optional[CameraInfo] = None
        
        self._image_left_count = 0
        self._image_right_count = 0
        self._inspvax_count = 0
        self._pointcloud_count = 0
    
    def add_inspvax(self, msg: INSPVAX):
        self.latest_inspvax = msg
        self._inspvax_count += 1
        
        if not self.initialized:
            self.lat_ref = msg.latitude
            self.lon_ref = msg.longitude
            self.initialized = True
            rospy.loginfo(f"参考点初始化：lat={self.lat_ref:.6f}, lon={self.lon_ref:.6f}")
        
        x_enu, y_enu = gps_to_enu(msg.latitude, msg.longitude, self.lat_ref, self.lon_ref)
        yaw = azimuth_to_yaw(msg.azimuth)
        
        self.global_pose_buffer.append([x_enu, y_enu, yaw])
        
        v_north = msg.north_velocity
        v_east = msg.east_velocity
        self.velocity_enu_buffer.append([v_east, v_north])
        
        rospy.loginfo_throttle(
            1.0,
            "INSPVAX#%d lat=%.6f lon=%.6f yaw=%.3frad enu=(%.2f,%.2f) hist=%d/%d",
            self._inspvax_count, msg.latitude, msg.longitude, yaw,
            x_enu, y_enu, len(self.global_pose_buffer), self.history_length,
        )
    
    def add_image_left(self, image: Image):
        self.latest_image_left = image
        self._image_left_count += 1

    def add_image_right(self, image: Image):
        self.latest_image_right = image
        self._image_right_count += 1
    
    def add_camera_left_info(self, info: CameraInfo):
        if self.camera_left_info is None:
            self.camera_left_info = info

    def add_camera_right_info(self, info: CameraInfo):
        if self.camera_right_info is None:
            self.camera_right_info = info
    
    def add_pointcloud(self, pc: PointCloud2):
        self.latest_pointcloud = pc
        self._pointcloud_count += 1
    
    def is_ready(self) -> bool:
        return (
            len(self.global_pose_buffer) >= self.history_length and
            self.latest_image_left is not None and
            self.latest_image_right is not None and
            self.latest_inspvax is not None and
            self.latest_pointcloud is not None
        )
    
    def not_ready_reasons(self) -> List[str]:
        reasons = []
        if len(self.global_pose_buffer) < self.history_length:
            reasons.append(f"history={len(self.global_pose_buffer)}/{self.history_length}")
        if self.latest_image_left is None:
            reasons.append("image_left=none")
        if self.latest_image_right is None:
            reasons.append("image_right=none")
        if self.latest_inspvax is None:
            reasons.append("inspvax=none")
        if self.latest_pointcloud is None:
            reasons.append("pointcloud=none")
        return reasons
    
    def get_local_history_trajectory(self) -> np.ndarray:
        """获取局部坐标系历史轨迹 [4, 3]"""
        if len(self.global_pose_buffer) == 0:
            return np.zeros((self.history_length, 3), dtype=np.float32)
        global_poses = [list(p) for p in self.global_pose_buffer]
        return convert_history_to_local(global_poses)
    
    def get_velocity_ego(self) -> np.ndarray:
        if len(self.velocity_enu_buffer) == 0 or self.latest_inspvax is None:
            return np.zeros(2, dtype=np.float32)
        
        v_east, v_north = self.velocity_enu_buffer[-1]
        current_yaw = azimuth_to_yaw(self.latest_inspvax.azimuth)
        
        vx = v_east * math.sin(current_yaw) - v_north * math.cos(current_yaw)
        vy = -v_east * math.cos(current_yaw) - v_north * math.sin(current_yaw)
        return np.array([vx, vy], dtype=np.float32)
    
    def get_acceleration_ego(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)
    
    def get_status_feature(self, command_one_hot: np.ndarray) -> np.ndarray:
        velocity = self.get_velocity_ego()
        acceleration = self.get_acceleration_ego()
        status = np.concatenate([command_one_hot, velocity, acceleration], dtype=np.float32)
        return status
    
    def clear(self):
        self.global_pose_buffer.clear()
        self.velocity_enu_buffer.clear()
        self.latest_inspvax = None
        self.latest_image = None
        self.latest_pointcloud = None
        self.initialized = False


# ==============================================================================
# 第六部分：显存监控
# ==============================================================================
class MemoryMonitor:
    """Jetson 显存监控工具"""
    def __init__(self):
        self.history = []
    
    def get_cuda_memory(self):
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0}
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return {
            'allocated': allocated / 1e6,
            'reserved': reserved / 1e6,
            'free': (total - reserved) / 1e6,
            'total': total / 1e6,
        }
    
    def get_system_memory(self):
        import psutil
        mem = psutil.virtual_memory()
        return {
            'used': mem.used / 1e6,
            'available': mem.available / 1e6,
            'percent': mem.percent,
        }
    
    def print_status(self, label=""):
        cuda_mem = self.get_cuda_memory()
        sys_mem = self.get_system_memory()
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}显存状态:")
        print(f"  CUDA: 已分配={cuda_mem['allocated']:.0f}MB, 预留={cuda_mem['reserved']:.0f}MB, 空闲={cuda_mem['free']:.0f}MB")
        print(f"  系统：已用={sys_mem['used']:.0f}MB, 可用={sys_mem['available']:.0f}MB ({sys_mem['percent']}%)")
    
    def check_threshold(self, threshold_percent=85):
        cuda_mem = self.get_cuda_memory()
        usage_percent = (cuda_mem['allocated'] / cuda_mem['total']) * 100
        if usage_percent > threshold_percent:
            rospy.logwarn(f"显存使用率过高：{usage_percent:.1f}%")
            torch.cuda.empty_cache()
            return False
        return True


# ==============================================================================
# 第七部分：ROS 节点主类
# ==============================================================================
class DiffusionDriveROSNode:
    """DiffusionDrive ROS1 节点"""
    
    def __init__(self):
        rospy.init_node('diffusiondrive_node', anonymous=False)
        self.config = DiffusionDriveROSConfig()
        self.model_config = TransfuserConfig()
        self.verbose_info = bool(rospy.get_param("~verbose_info", True))
        
        self.buffer = DataBuffer(history_length=self.config.history_length)

        # 模型组件
        self.agent = None

        # 缓存 ToTensor transform，避免每帧重建
        self._to_tensor = transforms.ToTensor()
        
        # 发布器
        self.trajectory_pub = rospy.Publisher(
            self.config.trajectory_pub_topic, PoseArray, queue_size=10
        )
        self.path_pub = rospy.Publisher(
            self.config.path_pub_topic, Path, queue_size=10
        )
        
        # 订阅器
        self._setup_subscribers()
        
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.print_status("节点初始化前")
        
        # 加载模型
        self._load_models()
        
        self.memory_monitor.print_status("模型加载后")
        
        # 推理频率控制
        self.inference_rate = rospy.Rate(self.config.inference_rate)
        self.last_infer_ms: Optional[float] = None
        self.inference_count = 0
        
        rospy.loginfo("DiffusionDrive ROS Node 初始化完成")
    
    def _setup_subscribers(self):
        rospy.Subscriber(self.config.inspvax_topic, INSPVAX, self._inspvax_callback, queue_size=1)
        rospy.Subscriber(self.config.image_left_topic, Image, self._image_left_callback, queue_size=1)
        rospy.Subscriber(self.config.image_right_topic, Image, self._image_right_callback, queue_size=1)
        rospy.Subscriber(self.config.camera_left_info_topic, CameraInfo, self._camera_left_info_callback, queue_size=1)
        rospy.Subscriber(self.config.camera_right_info_topic, CameraInfo, self._camera_right_info_callback, queue_size=1)
        rospy.Subscriber(self.config.pointcloud_topic, PointCloud2, self._pointcloud_callback, queue_size=1)
        rospy.loginfo(f"已订阅：{self.config.inspvax_topic}, {self.config.image_left_topic}, {self.config.image_right_topic}, {self.config.pointcloud_topic}")
    
    def _inspvax_callback(self, msg: INSPVAX):
        self.buffer.add_inspvax(msg)
    
    def _image_left_callback(self, msg: Image):
        self.buffer.add_image_left(msg)
    
    def _image_right_callback(self, msg: Image):
        self.buffer.add_image_right(msg)
    
    def _camera_left_info_callback(self, msg: CameraInfo):
        self.buffer.add_camera_left_info(msg)

    def _camera_right_info_callback(self, msg: CameraInfo):
        self.buffer.add_camera_right_info(msg)
    
    def _pointcloud_callback(self, msg: PointCloud2):
        self.buffer.add_pointcloud(msg)
    
    def _load_models(self):
        rospy.loginfo("正在加载 DiffusionDrive 模型...")
        try:
            self.agent = TransfuserAgent(
                config=self.model_config,
                lr=0.0,
                checkpoint_path=self.config.checkpoint
            )
            self.agent = self.agent.to(self.config.device)
            self.agent.eval()
            rospy.loginfo(f"模型加载完成，设备={self.config.device}")
            self._warmup_inference()
        except Exception as e:
            rospy.logerr(f"模型加载失败：{e}")
            import traceback
            traceback.print_exc()
            raise

    def _warmup_inference(self):
        """CUDA 推理预热，消除 JIT 编译和内存分配延迟"""
        if self.config.device != "cuda":
            return
        rospy.loginfo("正在预热 CUDA 推理（3次 dummy forward）...")
        status_dim = 4 + 2 + 2  # command_one_hot + velocity + acceleration
        dummy = {
            'camera_feature': torch.zeros(1, 3, 256, 1024, device=self.config.device),
            'lidar_feature': torch.zeros(1, 1, 256, 256, device=self.config.device),
            'status_feature': torch.zeros(1, status_dim, device=self.config.device),
        }
        with torch.inference_mode():
            for _ in range(3):
                self.agent.forward(dummy)
        torch.cuda.synchronize()
        rospy.loginfo("CUDA 推理预热完成")
    
    def _decode_image(self, ros_image: Image) -> np.ndarray:
        """解码 sensor_msgs/Image 为 numpy 数组"""
        if ros_image.encoding == 'rgb8':
            img_np = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                ros_image.height, ros_image.width, 3
            )
        elif ros_image.encoding == 'bgr8':
            img_np = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                ros_image.height, ros_image.width, 3
            )
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        elif ros_image.encoding == 'mono8':
            img_np = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(
                ros_image.height, ros_image.width
            )
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_np = np.frombuffer(ros_image.data, dtype=np.uint8)
            if img_np.size == ros_image.height * ros_image.width * 3:
                img_np = img_np.reshape(ros_image.height, ros_image.width, 3)
                if ros_image.encoding.startswith('bgr'):
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"不支持的图像编码：{ros_image.encoding}")
        return img_np
    
    def _parse_pointcloud2(self, pc_msg: PointCloud2) -> np.ndarray:
        """
        解析 sensor_msgs/PointCloud2 为 numpy 数组 (N, 4) [x, y, z, intensity]
        向量化实现：避免 Python for 循环，使用 numpy structured view
        """
        fields = {field.name: field for field in pc_msg.fields}

        for field_name in ['x', 'y', 'z']:
            if field_name not in fields:
                rospy.logwarn(f"点云缺少字段：{field_name}")
                return np.zeros((0, 4), dtype=np.float32)

        num_points = pc_msg.width * pc_msg.height
        point_step = pc_msg.point_step

        # 一次性读入所有字节，reshape 为 (N, point_step)
        raw = np.frombuffer(pc_msg.data, dtype=np.uint8).reshape(num_points, point_step)

        x_offset = fields['x'].offset
        y_offset = fields['y'].offset
        z_offset = fields['z'].offset
        intensity_field = fields.get('intensity', fields.get('i'))
        intensity_offset = intensity_field.offset if intensity_field is not None else -1

        # 向量化提取：切片后 ascontiguousarray 再 view 为 float32
        x = np.ascontiguousarray(raw[:, x_offset:x_offset + 4]).view(np.float32).ravel()
        y = np.ascontiguousarray(raw[:, y_offset:y_offset + 4]).view(np.float32).ravel()
        z = np.ascontiguousarray(raw[:, z_offset:z_offset + 4]).view(np.float32).ravel()
        if intensity_offset >= 0:
            intensity = np.ascontiguousarray(
                raw[:, intensity_offset:intensity_offset + 4]
            ).view(np.float32).ravel()
        else:
            intensity = np.zeros(num_points, dtype=np.float32)

        points = np.stack([x, y, z, intensity], axis=1)
        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        return points[valid_mask]
    
    def _build_lidar_feature(self, raw_lidar) -> torch.Tensor:
        """
        Build LiDAR BEV histogram feature from ROS PointCloud2 message.
        
        参考 diffusiondrive_b2d_agent.py._build_lidar_feature
        
        Coordinate conversion chain:
        ROS PointCloud2 (sensor frame)
          → CARLA raw (X-fwd, Y-right, Z-up) via X,Y swap
          → NuScenes LiDAR (X-right, Y-fwd, Z-up)
          → Ego (X-fwd, Y-left, Z-up) via lidar2ego matrix
        
        Returns:
            torch.Tensor: LiDAR feature of shape (1, 256, 256)
        """
        # 1. 解析点云 (N, 4) [x, y, z, intensity]  
        lidar_pc = raw_lidar[:, :3].copy()  # (N, 3) xyz
        
        # 2. 坐标转换：CARLA raw → NuScenes LiDAR → Ego
        # 交换 X 和 Y (CARLA raw → NuScenes)
        N = lidar_pc.shape[0]
        lidar_pc[:, 0], lidar_pc[:, 1] = raw_lidar[:, 1].copy(), raw_lidar[:, 0].copy()
        
        # NuScenes → Ego (使用齐次坐标)
        homo = np.hstack([lidar_pc, np.ones((N, 1))])  # (N, 4)
        lidar_pc = (self.config.lidar2ego @ homo.T).T[:, :3]  # (N, 3) in ego frame
        
        # 3. 高度滤波
        lidar_pc = lidar_pc[lidar_pc[:, 2] < self.model_config.max_height_lidar]
        above = lidar_pc[lidar_pc[:, 2] > self.model_config.lidar_split_height]

        # 4. 构建 2D 直方图
        xbins = np.linspace(
            self.model_config.lidar_min_x, self.model_config.lidar_max_x,
            int((self.model_config.lidar_max_x - self.model_config.lidar_min_x) * int(self.model_config.pixels_per_meter) + 1),
        )
        ybins = np.linspace(
            self.model_config.lidar_min_y, self.model_config.lidar_max_y,
            int((self.model_config.lidar_max_y - self.model_config.lidar_min_y) * int(self.model_config.pixels_per_meter) + 1),
        )
        
        hist = np.histogramdd(above[:, :2], bins=(xbins, ybins))[0]
        hist = np.clip(hist, 0, self.model_config.hist_max_per_pixel)
        overhead_splat = hist / self.model_config.hist_max_per_pixel
        
        # 5. 转换为 (1, 256, 256) 格式
        features = np.stack([overhead_splat], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        
        return torch.tensor(features)

    def _build_camera_feature(self, image_left_np: np.ndarray, image_right_np: np.ndarray) -> torch.Tensor:
        """
        构建相机特征 - 使用左右前视摄像头拼接
        
        处理流程:
        1. 水平拼接 (left + right)并裁剪到(3840,960)
        2. 调整大小到 (1024, 256)
        3. 归一化到 [0, 1]
        4. 转换为 (3, 256, 1024)
        
        Args:
            image_left_np: (1080, 1920, 3) 左前视图像
            image_right_np: (1080, 1920, 3) 右前视图像
            
        Returns:
            camera_feature: (3, 256, 1024) torch.Tensor
        """
        crop_h_start = self.config.crop_h_start
        crop_h_end = self.config.crop_h_end
        left_cropped = image_left_np[crop_h_start:crop_h_end]
        right_cropped = image_right_np[crop_h_start:crop_h_end]

        stitched_image = np.concatenate([left_cropped, right_cropped], axis=1)
        resized_image = cv2.resize(stitched_image, (self.config.camera_feature_width, self.config.camera_feature_height))
        tensor_image = self._to_tensor(resized_image)
        
        return tensor_image
    
    @staticmethod
    def command_to_onehot(carla_command: int, expand: bool = False) -> np.ndarray:
        if expand:
            cmd = np.zeros(6, dtype=np.float32)
            mapping = {1: 0, 3: 1, 2: 2, 4: 1, 5: 4, 6: 5}
            if carla_command in mapping:
                cmd[mapping[carla_command]] = 1.0
            else:
                cmd[1] = 1.0
            return cmd
        else:
            cmd = np.zeros(4, dtype=np.float32)
            if carla_command == 1:
                cmd[0] = 1.0
            elif carla_command == 3:
                cmd[1] = 1.0
            elif carla_command == 2:
                cmd[2] = 1.0
            else:
                cmd[1] = 1.0
            return cmd
    
    def _infer(self, command_one_hot: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        
        image_left_np = self._decode_image(self.buffer.latest_image_left)
        image_right_np = self._decode_image(self.buffer.latest_image_right)
        raw_lidar = self._parse_pointcloud2(self.buffer.latest_pointcloud)
        t1 = time.perf_counter()
        
        camera_feature = self._build_camera_feature(image_left_np, image_right_np)
        lidar_feature = self._build_lidar_feature(raw_lidar)
        status_feature = self.buffer.get_status_feature(
            np.array(self.config.command_one_hot, dtype=np.float32)
        )
        
        features = {
            'camera_feature': camera_feature.unsqueeze(0).to(self.config.device),
            'lidar_feature': lidar_feature.unsqueeze(0).to(self.config.device),
            'status_feature': torch.from_numpy(status_feature).float().unsqueeze(0).to(self.config.device),
        }
        t2 = time.perf_counter()

        with torch.inference_mode():
            predictions = self.agent.forward(features)
        t3 = time.perf_counter()
        
        if "trajectory" in predictions:
            pred_traj = predictions["trajectory"]
        elif "pred_traj" in predictions:
            pred_traj = predictions["pred_traj"]
        else:
            raise KeyError(f"无法找到轨迹输出，可用键：{list(predictions.keys())}")
        
        pred_traj_np = pred_traj.squeeze(0).cpu().numpy()
        
        self.last_infer_ms = (t3 - t0) * 1000.0
        self.inference_count += 1
        
        if self.verbose_info:
            print(f"[Timing] 信息解码：{(t1-t0)*1000:.1f}ms | "
                  f"特征构建: {(t2-t1)*1000:.1f}ms | "
                  f"模型推理：{(t3-t2)*1000:.1f}ms | "
                  f"总计：{(t3-t0)*1000:.1f}ms")
            pred_traj_str = np.array2string(
                pred_traj_np,
                precision=3,
                suppress_small=True,
                separator=", ",
            )
            rospy.loginfo(
                "Infer done: %.1fms pred_traj_np(shape=%s)=%s",
                float(self.last_infer_ms),
                tuple(pred_traj_np.shape),
                pred_traj_str,
            )
        
        return pred_traj_np
    
    def _transform_to_world(self, traj_ego: np.ndarray) -> np.ndarray:
        if self.buffer.latest_inspvax is None or len(self.buffer.global_pose_buffer) == 0:
            return traj_ego
        
        inspvax = self.buffer.latest_inspvax
        x_enu, y_enu = gps_to_enu(
            inspvax.latitude, inspvax.longitude,
            self.buffer.lat_ref, self.buffer.lon_ref,
        )
        z_enu = inspvax.height
        yaw = azimuth_to_yaw(inspvax.azimuth)
        
        cos_h = math.cos(yaw)
        sin_h = math.sin(yaw)
        
        traj_world = np.zeros_like(traj_ego)
        for i in range(traj_ego.shape[0]):
            local_x, local_y = traj_ego[i, 0], traj_ego[i, 1]
            
            dx = local_x * sin_h - local_y * cos_h
            dy = -local_x * cos_h - local_y * sin_h
            
            traj_world[i, 0] = x_enu + dx
            traj_world[i, 1] = y_enu + dy
            traj_world[i, 2] = z_enu if traj_ego.shape[1] > 2 else 0
            
            if traj_ego.shape[1] > 3:
                traj_world[i, 3] = (yaw + traj_ego[i, 2] + math.pi) % (2 * math.pi) - math.pi
        
        return traj_world
    
    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
    
    def _publish_trajectory(self, traj_world: np.ndarray):
        stamp = rospy.Time.now()
        frame_id = "enu"
        
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = frame_id
        
        for pt in traj_world:
            pose = PoseStamped()
            pose.header = pose_array.header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2]) if len(pt) > 2 else 0.0
            heading = float(pt[3]) if len(pt) > 3 else 0.0
            pose.pose.orientation = self._yaw_to_quaternion(heading)
            pose_array.poses.append(pose.pose)
        
        self.trajectory_pub.publish(pose_array)
        
        path = Path()
        path.header = pose_array.header
        path.poses = [
            PoseStamped(header=pose_array.header, pose=pose)
            for pose in pose_array.poses
        ]
        self.path_pub.publish(path)
    
    def run(self):
        rospy.loginfo("开始 DiffusionDrive 推理循环...")
        while not rospy.is_shutdown():
            try:
                if self.buffer.is_ready():
                    command_one_hot = self.command_to_onehot(carla_command=3, expand=False)
                    traj_ego = self._infer(command_one_hot)
                    traj_world = self._transform_to_world(traj_ego)
                    self._publish_trajectory(traj_world)
                    self.memory_monitor.check_threshold(threshold_percent=85)
                else:
                    rospy.loginfo_throttle(
                        5,
                        "等待数据收集中... %s",
                        ", ".join(self.buffer.not_ready_reasons()) or "unknown"
                    )
                
                self.inference_rate.sleep()
            except Exception as e:
                rospy.logerr(f"推理错误：{e}")
                import traceback
                traceback.print_exc()
                self.inference_rate.sleep()
    
    def shutdown(self):
        rospy.loginfo(f"节点关闭，共完成 {self.inference_count} 次推理")


# ==============================================================================
# 第八部分：主函数
# ==============================================================================
def main():
    try:
        node = DiffusionDriveROSNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点启动失败：{e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    exit(main())