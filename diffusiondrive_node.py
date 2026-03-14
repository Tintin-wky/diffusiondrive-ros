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
# 第一部分：修复 torch.distributed
# ==============================================================================
import torch
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
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField, Imu
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, Twist
from novatel_oem7_msgs.msg import INSPVAX  # NovAtel 惯导消息

# TransFuser/DiffusionDrive 导入
from model.transfuser_agent import TransfuserAgent
from model.transfuser_config import TransfuserConfig
from model.transfuser_features import TransfuserFeatureBuilder
from model.local_nuplan import TrajectorySampling

# ==============================================================================
# 第三部分：配置类
# ==============================================================================
class DiffusionDriveROSConfig:
    """ROS 节点配置 - 适配 TransFuser 模型"""
    def __init__(self):
        self.verbose_info = bool(rospy.get_param("~verbose_info", True))
        # ========== 模型路径 ==========
        self.checkpoint = '/root/diffusiondrive/ckpts/diffusiondrive_navsim_88p1_PDMS.pth'
        
        # ========== 模型配置 ==========
        self.action_horizon = 8
        self.action_dim = 3  # [x, y, heading]
        self.trajectory_sampling = TrajectorySampling(time_horizon=2, interval_length=0.5)
        self.command_one_hot = [0, 1, 0, 0]  # 默认直行 [左，直，右，其他]
        
        # ========== ROS Topic 配置 ==========
        self.inspvax_topic = "/bynav/inspvax"
        self.image_left_topic = "/camera/left_front_view"      # 左前视
        self.image_right_topic = "/camera/right_front_view"    # 右前视
        self.camera_left_info_topic = "/camera/left_front_view/camera_info"
        self.camera_right_info_topic = "/camera/right_front_view/camera_info"
        self.pointcloud_topic = "/rslidar_points"
        self.imu_topic = "/gps/imu"  # IMU 传感器（加速度）
        self.stitched_image_pub_topic = "/diffusiondrive/stitched_image"  # 拼接图像
        self.trajectory_pub_topic = "/diffusiondrive/trajectory"  # LiDAR frame 轨迹路径
        self.trajectory_points_pub_topic = "/diffusiondrive/trajectory_points"  # LiDAR frame 轨迹点
        self.history_trajectory_pub_topic = "/diffusiondrive/history_trajectory"  # 本地历史轨迹路径

        # ========== 推理配置 ==========
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_rate = 10  # Hz
        
        # ========== 激光雷达配置 (参考 diffusiondrive_b2d_agent.py) ==========
        # ego → NuScenes LiDAR 转换矩阵 (lidar2ego 的逆)
        self.ego2lidar = np.array([
            [ 1.,  0.,  0.,  0.  ],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  1., -1.],
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
            [ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.  ],
            [ 0.,  0.,  1.,  1.],
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
        
        # ========== 图像配置 ==========
        #  (1920x1080 → 裁剪后缩放)
        self.image_width = 1920
        self.image_height = 1080

        # 左摄像头：裁剪参数
        self.left_crop_h_start = 0
        self.left_crop_h_end = 1080-48-172
        self.left_crop_w_start = 0
        self.left_crop_w_end = 1920-200

        # 右摄像头：裁剪参数
        self.right_crop_h_start = 48
        self.right_crop_h_end = 1080-172
        self.right_crop_w_start = 200
        self.right_crop_w_end = 1920

        # 特征输出配置
        self.camera_feature_width = 1024
        self.camera_feature_height = 256


# ==============================================================================
# 第四部分：坐标转换工具函数
# ==============================================================================
def gps_to_enu(lat: float, lon: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    """
    GPS (lat/lon) → 局部 ENU 坐标 (近似公式)
    适用于小范围 (例如半径 10km 以内)
    """
    R = 6378137.0  # 地球半径 (米)
    
    # 将角度转换为弧度
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat_ref_rad = math.radians(lat_ref)
    lon_ref_rad = math.radians(lon_ref)
    
    # 计算差值
    d_lat = lat_rad - lat_ref_rad
    d_lon = lon_rad - lon_ref_rad
    
    # ENU 坐标计算
    # x (East): 经度差 * 半径 * 参考纬度的余弦 (修正经线收敛)
    x = d_lon * R * math.cos(lat_ref_rad)
    
    # y (North): 纬度差 * 半径
    y = d_lat * R
    
    return x, y

def azimuth_to_yaw(azimuth_deg: float) -> float:
    """INSPVAX azimuth → 模型 yaw"""
    azimuth_rad = math.radians(azimuth_deg)
    yaw = math.pi / 2 - azimuth_rad
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return yaw


def enu_to_ego(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    """ENU 位移 → 自车局部坐标系 (X-fwd, Y-left)"""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    return local_x, local_y


def convert_history_to_local(global_poses: List[Tuple[float, float, float]],
                            current_pose: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    将全球坐标系历史轨迹转换到当前ego坐标系

    Args:
        global_poses: 全局坐标系下的历史位姿列表 [(x, y, yaw), ...]
        current_pose: 当前位姿 (x, y, yaw)，如果为None则使用缓冲区最后一个位姿

    Returns:
        np.ndarray: (N, 3) 局部坐标系下的轨迹 [local_x, local_y, heading_local]
    """
    if current_pose is not None:
        x0, y0, yaw0 = current_pose
    else:
        x0, y0, yaw0 = poglobal_posesses[-1]

    local_poses = []
    for x, y, yaw in global_poses:
        dx = x - x0
        dy = y - y0
        local_x, local_y = enu_to_ego(dx, dy, yaw0)
        heading_local = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
        local_poses.append([local_x, local_y, heading_local])

    return np.array(local_poses, dtype=np.float32)

# ==============================================================================
# 第五部分：数据缓存类
# ==============================================================================
class DataBuffer:
    """ROS 数据缓存"""
    def __init__(self, trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=2, interval_length=0.5)):
        self.verbose_info = bool(rospy.get_param("~verbose_info", False))
        
        # 全局位姿缓冲更新间隔（0.1s）
        self._pose_buffer_update_interval = 0.1
        self._last_pose_buffer_update_time = 0.0
        self.trajectory_sampling = trajectory_sampling
        self.history_length = int(self.trajectory_sampling.time_horizon / self._pose_buffer_update_interval)
        self.global_pose_buffer = deque(maxlen=self.history_length)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)

        self.latest_inspvax: Optional[INSPVAX] = None
        self.latest_image_left: Optional[Image] = None
        self.latest_image_right: Optional[Image] = None
        self.latest_pointcloud: Optional[PointCloud2] = None
        self.latest_imu: Optional[Imu] = None

        self.lat_ref = None
        self.lon_ref = None
        self.initialized = False
        self.camera_left_info: Optional[CameraInfo] = None
        self.camera_right_info: Optional[CameraInfo] = None

        self._image_left_count = 0
        self._image_right_count = 0
        self._inspvax_count = 0
        self._pointcloud_count = 0
        self._imu_count = 0


        # IMU 加速度滤波（移动平均窗口大小，120Hz采样 -> 10个样本约83ms）
        self._imu_accel_buffer = deque(maxlen=10)  # 保留最近 10 个 IMU 样本用于低通滤波
        self._imu_filter_alpha = 0.15  # 一阶低通滤波系数 (0-1, 越小越平滑)
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)
    
    def add_inspvax(self, msg: INSPVAX):
        self.latest_inspvax = msg
        self._inspvax_count += 1

        if not self.initialized:
            self.lat_ref = msg.latitude
            self.lon_ref = msg.longitude
            self.initialized = True
            self._last_pose_buffer_update_time = rospy.get_time()
            rospy.loginfo(f"参考点初始化：lat={self.lat_ref:.6f}, lon={self.lon_ref:.6f}")

        x_enu, y_enu = gps_to_enu(msg.latitude, msg.longitude, self.lat_ref, self.lon_ref)
        yaw = azimuth_to_yaw(msg.azimuth)

        # 全局位姿缓冲以 0.1s 频率更新（高频精细采样）
        current_time = rospy.get_time()
        if current_time - self._last_pose_buffer_update_time >= self._pose_buffer_update_interval:
            self.global_pose_buffer.append([x_enu, y_enu, yaw])
            self._last_pose_buffer_update_time = current_time


        vx, vy = enu_to_ego(msg.east_velocity, msg.north_velocity, yaw)
        self.velocity = np.array([vx, vy], dtype=np.float32)


        if self.verbose_info:
            ax = self.acceleration[0]
            ay = self.acceleration[1]
            rospy.loginfo_throttle(
                1.0,
                "INSPVAX#%d enu=(%.2f,%.2f) yaw=%.3frad vel_ego=(%.2f,%.2f) accel_ego=(%.2f,%.2f)",
                self._inspvax_count, x_enu, y_enu, yaw, vx, vy, ax, ay
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

    def add_imu(self, msg: Imu):
        """
        处理 IMU 消息，提取加速度并进行滤波

        IMU 消息中的加速度坐标系已与 ego 对齐（前X左Y）
        应用两种滤波：
        1. 移动平均：保留最近 10 个样本的滑动窗口（~83ms @ 120Hz）
        2. 一阶低通滤波：在当前滤波值和新值之间进行指数加权平均
        """
        self.latest_imu = msg
        self._imu_count += 1

        # 提取加速度（ego frame: x-forward, y-left）
        raw_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y], dtype=np.float32)

        # 方法1：移动平均滤波
        self._imu_accel_buffer.append(raw_accel)
        if len(self._imu_accel_buffer) > 0:
            smoothed_accel = np.mean(list(self._imu_accel_buffer), axis=0)
        else:
            smoothed_accel = raw_accel

        # 方法2：一阶低通滤波（指数加权平均）
        # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        # alpha 越小越平滑，越大越跟踪敏捷
        self.acceleration = (self._imu_filter_alpha * smoothed_accel +
                               (1.0 - self._imu_filter_alpha) * self.acceleration)

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
        """
        获取局部坐标系历史轨迹

        采样策略：
        - 全局位姿缓冲以 0.1s 频率存储（maxlen=20，共 2s 历史）
        - 以 0.5s 间隔（5 帧）提取 4 个历史位姿
        - 使用缓冲区最后一帧作为当前位姿（参考点）

        Returns:
            np.ndarray: (4, 3) 局部坐标系下的历史轨迹 [local_x, local_y, heading_local]
                       对应 t-1.5s, t-1.0s, t-0.5s, t (当前)
        """
        current_pose = self.global_pose_buffer[-1]
        buffer_len = len(self.global_pose_buffer)
        interval = int(self.trajectory_sampling.interval_length / self._pose_buffer_update_interval)
        assert buffer_len >= interval * (self.trajectory_sampling.num_poses - 1) + 1

        # 以 0.5s 间隔（5 帧）向后提取 4 个历史位姿
        # 索引：[t-1.5s, t-1.0s, t-0.5s, t]
        # 对应：[n-16, n-11, n-6, n-1]
        sampling_indices = []
        for i in range(self.trajectory_sampling.num_poses):
            idx = (i + 1) * interval - 1
            sampling_indices.append(idx)

        # 构建历史位姿列表
        global_poses = []
        for idx in sampling_indices:
            global_poses.append(list(self.global_pose_buffer[idx]))

        return convert_history_to_local(global_poses, current_pose)
    
    def get_status_feature(self, command_one_hot: np.ndarray) -> np.ndarray:
        return np.concatenate([command_one_hot, self.velocity, self.acceleration], dtype=np.float32)


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
        """打印当前显存状态"""
        cuda_mem = self.get_cuda_memory()
        sys_mem = self.get_system_memory()
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}显存状态:")
        print(f"  CUDA: 已分配={cuda_mem['allocated']:.0f}MB, "
              f"预留={cuda_mem['reserved']:.0f}MB, "
              f"空闲={cuda_mem['free']:.0f}MB/{cuda_mem['total']:.0f}MB")
        print(f"  系统：已用={sys_mem['used']:.0f}MB, "
              f"可用={sys_mem['available']:.0f}MB ({sys_mem['percent']}%)")
        
        self.history.append({
            'label': label,
            'cuda': cuda_mem,
            'sys': sys_mem,
            'time': rospy.Time.now()
        })
    
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
        
        self.buffer = DataBuffer(self.config.trajectory_sampling)
        self.buffer.verbose_info = self.config.verbose_info

        # 模型组件
        self.agent = None

        # 缓存 ToTensor transform，避免每帧重建
        self._to_tensor = transforms.ToTensor()
        
        # 发布器
        self.stitched_image_pub = rospy.Publisher(
            self.config.stitched_image_pub_topic, Image, queue_size=10
        )
        self.trajectory_pub = rospy.Publisher(
            self.config.trajectory_pub_topic, Path, queue_size=10
        )
        self.trajectory_points_pub = rospy.Publisher(
            self.config.trajectory_points_pub_topic, PoseArray, queue_size=10
        )
        self.history_trajectory_pub = rospy.Publisher(
            self.config.history_trajectory_pub_topic, Path, queue_size=10
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

        if self.config.verbose_info:
            # ========== 打印配置 ==========
            print("=" * 60)
            print("DiffusionDrive ROS 配置")
            print("=" * 60)
            print(f"设备：{self.config.device}")
            print(f"推理频率：{self.inference_rate} Hz")
            print(f"BEV 范围：[{self.config.lidar_min_x}, {self.config.lidar_max_x}] x [{self.config.lidar_min_y}, {self.config.lidar_max_y}]")
            print(f"BEV 分辨率：{int((self.config.lidar_max_x - self.config.lidar_min_x) * self.config.pixels_per_meter)}x"
                f"{int((self.config.lidar_max_y - self.config.lidar_min_y) * self.config.pixels_per_meter)}")
            print("=" * 60)
        
        rospy.loginfo("DiffusionDrive ROS Node 初始化完成")
    
    def _setup_subscribers(self):
        rospy.Subscriber(self.config.inspvax_topic, INSPVAX, self._inspvax_callback, queue_size=1)
        rospy.Subscriber(self.config.image_left_topic, Image, self._image_left_callback, queue_size=1)
        rospy.Subscriber(self.config.image_right_topic, Image, self._image_right_callback, queue_size=1)
        rospy.Subscriber(self.config.camera_left_info_topic, CameraInfo, self._camera_left_info_callback, queue_size=1)
        rospy.Subscriber(self.config.camera_right_info_topic, CameraInfo, self._camera_right_info_callback, queue_size=1)
        rospy.Subscriber(self.config.pointcloud_topic, PointCloud2, self._pointcloud_callback, queue_size=1)
        rospy.Subscriber(self.config.imu_topic, Imu, self._imu_callback, queue_size=10)
        rospy.loginfo(f"已订阅：{self.config.inspvax_topic}, {self.config.image_left_topic}, {self.config.image_right_topic}, {self.config.pointcloud_topic}, {self.config.imu_topic}")
    
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

    def _imu_callback(self, msg: Imu):
        self.buffer.add_imu(msg)

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

    def _numpy_to_ros_image(self, img_np: np.ndarray, encoding: str = 'rgb8', frame_id: str = 'camera') -> Image:
        """将 numpy 数组转换为 sensor_msgs/Image"""
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.height, msg.width = img_np.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = img_np.nbytes // img_np.shape[0]
        msg.data = img_np.tobytes()
        return msg
    
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
        
        Returns:
            torch.Tensor: LiDAR feature of shape (1, 256, 256)
        """
        # 1. 解析点云 (N, 4) [x, y, z, intensity]  
        lidar_pc = raw_lidar[:, :3].copy()  # (N, 3) xyz
        
        # 2. 坐标转换：LiDAR → Ego
        N = lidar_pc.shape[0]
        
        # 使用齐次坐标
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

    def _build_camera_feature(self, image_left_np: np.ndarray, image_right_np: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        构建相机特征 - 使用左右前视摄像头拼接

        处理流程:
        1. 对左右图分别应用各自的裁剪参数
        2. 水平拼接 (left + right)
        3. 调整大小到 (1024, 256)
        4. 归一化到 [0, 1]
        5. 转换为 (3, 256, 1024)

        Args:
            image_left_np: (1080, 1920, 3) 左前视图像
            image_right_np: (1080, 1920, 3) 右前视图像

        Returns:
            camera_feature: (3, 256, 1024) torch.Tensor
            stitched_image: (H, W, 3) np.ndarray 原始拼接图像
        """
        # 应用左摄像头裁剪参数
        left_cropped = image_left_np[
            self.config.left_crop_h_start:self.config.left_crop_h_end,
            self.config.left_crop_w_start:self.config.left_crop_w_end
        ]

        # 应用右摄像头裁剪参数
        right_cropped = image_right_np[
            self.config.right_crop_h_start:self.config.right_crop_h_end,
            self.config.right_crop_w_start:self.config.right_crop_w_end
        ]

        # 如果高度不一致，调整为相同高度
        if left_cropped.shape[0] != right_cropped.shape[0]:
            min_h = min(left_cropped.shape[0], right_cropped.shape[0])
            left_cropped = left_cropped[:min_h]
            right_cropped = right_cropped[:min_h]

        # 水平拼接
        stitched_image = np.concatenate([left_cropped, right_cropped], axis=1)

        # 缩放到特征尺寸
        resized_image = cv2.resize(stitched_image, (self.config.camera_feature_width, self.config.camera_feature_height))
        tensor_image = self._to_tensor(resized_image)

        return tensor_image, resized_image
    
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

    def _transform_to_lidar_frame(self, traj_ego: np.ndarray) -> np.ndarray:
        """
        将 ego frame 的轨迹转换到 lidar frame

        使用 ego2lidar 转换矩阵（将 ego 坐标转换到 lidar）

        Args:
            traj_ego: (N, 3) ego frame 中的轨迹点 [x, y, z,]

        Returns:
            traj_lidar: (N, 3) lidar frame 中的轨迹点
        """
        N = traj_ego.shape[0]
        traj_lidar = np.zeros_like(traj_ego)

        # 转换位置 (x, y, z)
        for i in range(N):
            # 构建齐次坐标
            ego_pos = np.array([traj_ego[i, 0], traj_ego[i, 1],
                               traj_ego[i, 2] if traj_ego.shape[1] > 2 else 0.0, 1.0])

            # 应用 ego2lidar 转换
            lidar_pos = self.config.ego2lidar @ ego_pos

            traj_lidar[i, 0] = lidar_pos[0]
            traj_lidar[i, 1] = lidar_pos[1]
            if traj_ego.shape[1] > 2:
                traj_lidar[i, 2] = lidar_pos[2]

        return traj_lidar

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def _publish_stitched_image(self, stitched_image: np.ndarray):
        """发布拼接后的图像"""
        try:
            ros_image = self._numpy_to_ros_image(stitched_image, encoding='rgb8', frame_id='camera_front')
            self.stitched_image_pub.publish(ros_image)
        except Exception as e:
            rospy.logwarn(f"发布拼接图像失败：{e}")

    def _publish_history_path(self, local_history: np.ndarray):
        """
        发布本地历史轨迹为 Path 消息（在 rslidar frame 中）

        Args:
            local_history: (N, 3) numpy array, 每行为 [x, y, yaw] 在 ego frame 中
        """
        if self.config.verbose_info:
            rospy.loginfo_throttle(
                1.0,
                "LocalHistory:\n%s",
                np.array2string(local_history, precision=3, suppress_small=True)
            )
        stamp = rospy.Time.now()
        frame_id = "rslidar"

        # 将本地历史轨迹从 ego frame 转换到 lidar frame
        local_history = self._transform_to_lidar_frame(local_history)

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = frame_id

        for pt in local_history:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = float(pt[0])
            pose_stamped.pose.position.y = float(pt[1])
            pose_stamped.pose.position.z = float(pt[2]) if len(pt) > 2 else 0.0
            heading = float(pt[3]) if len(pt) > 3 else 0.0
            pose_stamped.pose.orientation = self._yaw_to_quaternion(heading)
            path.poses.append(pose_stamped)

        self.history_trajectory_pub.publish(path)


    def _publish_trajectory(self, traj_ego: np.ndarray):
        """
        发布 LiDAR frame 中的轨迹，包括路径和轨迹点

        发布两个消息：
        1. Path - 用于可视化连接的路径线
        2. PoseArray - 用于可视化离散的轨迹点
        """
        if self.config.verbose_info:
            rospy.loginfo_throttle(
                1.0,
                "Predict History:\n%s\n",
                np.array2string(traj_ego, precision=3, suppress_small=True)
            )
        stamp = rospy.Time.now()
        frame_id = "rslidar"

        # 转换到 lidar frame
        traj_ego = self._transform_to_lidar_frame(traj_ego)

        # 构建 Path 消息（连接的路径线）
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = frame_id

        # 构建 PoseArray 消息（轨迹点）
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = frame_id

        for pt in traj_ego:
            # 创建 PoseStamped 用于 Path
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = float(pt[0])
            pose_stamped.pose.position.y = float(pt[1])
            pose_stamped.pose.position.z = float(pt[2]) if len(pt) > 2 else 0.0
            heading = float(pt[3]) if len(pt) > 3 else 0.0
            pose_stamped.pose.orientation = self._yaw_to_quaternion(heading)
            path.poses.append(pose_stamped)

            # 创建 Pose 用于 PoseArray
            pose = pose_stamped.pose
            pose_array.poses.append(pose)

        # 同时发布 Path 和 PoseArray
        self.trajectory_pub.publish(path)
        self.trajectory_points_pub.publish(pose_array)
        

    def _infer(self, command_one_hot: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        image_left_np = self._decode_image(self.buffer.latest_image_left)
        image_right_np = self._decode_image(self.buffer.latest_image_right)
        raw_lidar = self._parse_pointcloud2(self.buffer.latest_pointcloud)
        t1 = time.perf_counter()

        camera_feature, stitched_image = self._build_camera_feature(image_left_np, image_right_np)
        lidar_feature = self._build_lidar_feature(raw_lidar)
        status_feature = self.buffer.get_status_feature(command_one_hot)

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

        # 发布拼接图像
        self._publish_stitched_image(stitched_image)

        self.last_infer_ms = (t3 - t0) * 1000.0
        self.inference_count += 1

        if self.config.verbose_info:
            print(f"[Timing] 信息解码：{(t1-t0)*1000:.1f}ms | "
                  f"特征构建: {(t2-t1)*1000:.1f}ms | "
                  f"模型推理：{(t3-t2)*1000:.1f}ms | "
                  f"总计：{(t3-t0)*1000:.1f}ms")
        return pred_traj_np

    def run(self):
        rospy.loginfo("开始 DiffusionDrive 推理循环...")
        while not rospy.is_shutdown():
            try:
                if self.buffer.is_ready():
                    command_one_hot = self.command_to_onehot(carla_command=3, expand=False)
                    traj_ego = self._infer(command_one_hot)
                    local_history = self.buffer.get_local_history_trajectory()
                    self._publish_history_path(local_history)
                    self._publish_trajectory(traj_ego)
                    # self.memory_monitor.check_threshold(threshold_percent=85)
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