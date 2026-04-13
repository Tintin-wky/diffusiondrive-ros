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
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField, Imu, CompressedImage
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, Twist, Point
from novatel_oem7_msgs.msg import INSPVAX  # NovAtel 惯导消息
from visualization_msgs.msg import Marker
from std_msgs.msg import Int32

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
        self.image_left_topic = "/camera/left_front_view/compressed"      # 左前视（压缩）
        self.image_right_topic = "/camera/right_front_view/compressed"    # 右前视（压缩）
        self.camera_left_info_topic = "/camera/left_front_view/camera_info"
        self.camera_right_info_topic = "/camera/right_front_view/camera_info"
        self.pointcloud_topic = "/rslidar_points"
        self.imu_topic = "/gps/imu"  # IMU 传感器（加速度）
        self.input_image_pub_topic = "/diffusiondrive/input_image"  # 输入图像（拼接/裁剪）
        self.fpv_trajectory_view_topic = "/diffusiondrive/fpv_trajectory_view"  # 第一视角轨迹可视化

        # ========== 图像输入模式 ==========
        # "dual_camera": 左右摄像头拼接 (默认)
        # "single_camera": 使用ZED
        self.image_input_mode = rospy.get_param("~image_input_mode", "single_camera")
        assert self.image_input_mode in ("dual_camera", "single_camera"), \
            f"image_input_mode 必须是 'dual_camera' 或 'single_camera', 得到: {self.image_input_mode}"
        self.image_single_camera_topic = "/camera/left/image_raw/compressed"
        self.history_marker_topic = "/diffusiondrive/history_marker"
        self.prediction_marker_topic = "/diffusiondrive/prediction_marker"
        self.command_topic = "/diffusiondrive/command"  # 导航命令输入

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

        self.clip_w = 224
        self.clip_h = 60
        self.final_w = self.image_width - self.clip_w
        self.final_h = self.final_w//2
        assert self.final_h < self.image_height

        # 左摄像头：裁剪参数
        self.left_crop_h_start = 0
        self.left_crop_h_end = self.final_h
        self.left_crop_w_end = self.image_width - self.clip_w
        self.left_crop_w_start = self.left_crop_w_end - self.final_w

        # 右摄像头：裁剪参数
        self.right_crop_h_start = self.clip_h
        self.right_crop_h_end = self.final_h + self.clip_h
        self.right_crop_w_start =  self.clip_w
        self.right_crop_w_end = self.final_w + self.right_crop_w_start

        # 单摄像头（ZED）：独立裁剪参数
        # 原始 1920x1080 → 裁剪后 1920x480
        self.single_crop_h_start = 280
        self.single_crop_h_end = self.image_width // 4 + self.single_crop_h_start
        self.single_crop_w_start = 0
        self.single_crop_w_end = self.image_width  # 保持全宽 1920

        # ========== 单相机投影参数（第一视角可视化） ==========
        # 相机内参
        self.single_camera_intrinsic = np.array([
            [1065.52408, 0.0, 982.56435],
            [0.0, 1067.06851, 540.12547],
            [0.0, 0.0, 1.0]
        ])
        self.single_camera_distortion = np.array([-0.041263, 0.017263, 0.002901, 0.001913])

        # rslidar → camera 外参（安装偏差）
        # 这些是相机相对于 lidar 的安装偏差（小角度）
        self.camera_to_rslidar_euler_deg = [-1.6125, -0.0436, -1.5416]  # degrees (安装偏差)
        self.camera_to_rslidar_translation = np.array([0.0105, -0.0451, -0.2600])  # meters (相机在lidar坐标系中的位置)

        # 坐标系轴转换：LiDAR (X前,Y左,Z上) → Camera (Z前,X右,Y下)
        # LiDAR X → Camera Z
        # LiDAR Y → Camera -X (左→右，取负)
        # LiDAR Z → Camera -Y (上→下，取负)
        self.lidar_to_camera_axis_transform = np.array([
            [ 0.0, -1.0,  0.0],  # Camera X = -LiDAR Y
            [ 0.0,  0.0, -1.0],  # Camera Y = -LiDAR Z
            [ 1.0,  0.0,  0.0],  # Camera Z = LiDAR X
        ])

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
    def __init__(self, trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=2, interval_length=0.5),
                 image_input_mode: str = "dual_camera"):
        self.verbose_info = bool(rospy.get_param("~verbose_info", False))
        self.image_input_mode = image_input_mode
        
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
        self.latest_image_single: Optional[CompressedImage] = None
        self.latest_pointcloud: Optional[PointCloud2] = None
        self.latest_imu: Optional[Imu] = None

        self.lat_ref = None
        self.lon_ref = None
        self.initialized = False
        self.camera_left_info: Optional[CameraInfo] = None
        self.camera_right_info: Optional[CameraInfo] = None

        self._image_left_count = 0
        self._image_right_count = 0
        self._image_single_count = 0
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
    
    def add_image_left(self, image: CompressedImage):
        self.latest_image_left = image
        self._image_left_count += 1

    def add_image_right(self, image: CompressedImage):
        self.latest_image_right = image
        self._image_right_count += 1

    def add_image_single(self, image: CompressedImage):
        self.latest_image_single = image
        self._image_single_count += 1

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
        # 单相机图像始终必需（用于第一视角可视化）
        images_ready = self.latest_image_single is not None
        # dual_camera 模式额外需要左右相机
        if self.image_input_mode == "dual_camera":
            images_ready = images_ready and (
                self.latest_image_left is not None and
                self.latest_image_right is not None
            )
        return (
            len(self.global_pose_buffer) >= self.history_length and
            images_ready and
            self.latest_inspvax is not None and
            self.latest_pointcloud is not None
        )

    def not_ready_reasons(self) -> List[str]:
        reasons = []
        if len(self.global_pose_buffer) < self.history_length:
            reasons.append(f"history={len(self.global_pose_buffer)}/{self.history_length}")
        # 单相机图像始终必需
        if self.latest_image_single is None:
            reasons.append("image_single=none")
        # dual_camera 模式额外需要左右相机
        if self.image_input_mode == "dual_camera":
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
        
        self.buffer = DataBuffer(self.config.trajectory_sampling, self.config.image_input_mode)
        self.buffer.verbose_info = self.config.verbose_info

        # 模型组件
        self.agent = None

        # 导航命令 (默认直行)
        # 1=左转, 2=右转, 3=直行, 其他=直行
        self.current_command = 3

        # 缓存 ToTensor transform，避免每帧重建
        self._to_tensor = transforms.ToTensor()
        
        # 发布器
        self.input_image_pub = rospy.Publisher(
            self.config.input_image_pub_topic, Image, queue_size=10
        )
        # Marker 发布器 (替代 Path/PoseArray)
        self.history_marker_pub = rospy.Publisher(
            self.config.history_marker_topic, Marker, queue_size=10
        )
        self.prediction_marker_pub = rospy.Publisher(
            self.config.prediction_marker_topic, Marker, queue_size=10
        )

        # 第一视角轨迹可视化发布器
        self.fpv_trajectory_pub = rospy.Publisher(
            self.config.fpv_trajectory_view_topic, Image, queue_size=10
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
        rospy.Subscriber(self.config.pointcloud_topic, PointCloud2, self._pointcloud_callback, queue_size=1)
        rospy.Subscriber(self.config.imu_topic, Imu, self._imu_callback, queue_size=10)

        # 单相机始终订阅（用于第一视角轨迹可视化）
        rospy.Subscriber(self.config.image_single_camera_topic, CompressedImage, self._image_single_callback, queue_size=1)
        rospy.loginfo(f"单相机订阅: {self.config.image_single_camera_topic} (用于第一视角可视化)")

        # dual_camera 模式额外订阅左右相机
        if self.config.image_input_mode == "dual_camera":
            rospy.Subscriber(self.config.image_left_topic, CompressedImage, self._image_left_callback, queue_size=1)
            rospy.Subscriber(self.config.image_right_topic, CompressedImage, self._image_right_callback, queue_size=1)
            rospy.Subscriber(self.config.camera_left_info_topic, CameraInfo, self._camera_left_info_callback, queue_size=1)
            rospy.Subscriber(self.config.camera_right_info_topic, CameraInfo, self._camera_right_info_callback, queue_size=1)
            rospy.loginfo(f"双相机模式额外订阅: {self.config.image_left_topic}, {self.config.image_right_topic}")

        rospy.loginfo(f"已订阅: {self.config.inspvax_topic}, {self.config.pointcloud_topic}, {self.config.imu_topic}")

        # 导航命令订阅
        rospy.Subscriber(self.config.command_topic, Int32, self._command_callback, queue_size=1)
        rospy.loginfo(f"导航命令订阅: {self.config.command_topic} (1=左转, 2=右转, 3=直行)")

    def _command_callback(self, msg: Int32):
        """接收导航命令"""
        cmd = msg.data
        if cmd in (1, 2, 3):
            self.current_command = cmd
            cmd_name = {1: "左转", 2: "右转", 3: "直行"}
            rospy.loginfo(f"导航命令更新: {cmd} ({cmd_name.get(cmd, '未知')})")
        else:
            rospy.logwarn(f"无效命令: {cmd}, 使用默认直行(3)")
            self.current_command = 3

    def _inspvax_callback(self, msg: INSPVAX):
        self.buffer.add_inspvax(msg)
    
    def _image_left_callback(self, msg: Image):
        self.buffer.add_image_left(msg)
    
    def _image_right_callback(self, msg: Image):
        self.buffer.add_image_right(msg)

    def _image_single_callback(self, msg: CompressedImage):
        self.buffer.add_image_single(msg)

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
    
    def _decode_image(self, ros_image: CompressedImage) -> np.ndarray:
        """解码 sensor_msgs/CompressedImage 压缩格式为 numpy 数组"""
        nparr = np.frombuffer(ros_image.data, dtype=np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 读取为 BGR
        if img_np is None:
            raise ValueError("无法解码压缩图像数据")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 转换为 RGB
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
            input_image: (H, W, 3) np.ndarray 拼接后的输入图像
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
        input_image = np.concatenate([left_cropped, right_cropped], axis=1)

        # 缩放到特征尺寸
        resized_image = cv2.resize(input_image, (self.config.camera_feature_width, self.config.camera_feature_height))
        tensor_image = self._to_tensor(resized_image)

        return tensor_image, resized_image

    def _build_camera_feature_single(self, image_np: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        single_camera 模式: 裁剪单张图像并构建相机特征
        Args:
            image_np: (H, W, 3) 单张图像，预期 1920x1080

        Returns:
            camera_feature: (3, 256, 1024) torch.Tensor
            cropped_image: (H', W', 3) np.ndarray 裁剪后的图像 (1920x480)
        """
        cropped = image_np[
            self.config.single_crop_h_start:self.config.single_crop_h_end,
            self.config.single_crop_w_start:self.config.single_crop_w_end
        ]
        resized = cv2.resize(cropped, (self.config.camera_feature_width, self.config.camera_feature_height))
        return self._to_tensor(resized), resized

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

    def _transform_to_lidar_frame(self, traj_ego: np.ndarray, z_height: float = 0.0) -> np.ndarray:
        """
        将 ego frame 的轨迹转换到 lidar frame

        使用 ego2lidar 转换矩阵（将 ego 坐标转换到 lidar）

        Args:
            traj_ego: (N, 3) ego frame 中的轨迹点 [x, y, heading]（注意：第三列是 heading 不是 z）
            z_height: 轨迹点的 z 坐标高度，默认 0.0（地面）

        Returns:
            traj_lidar: (N, 3) lidar frame 中的轨迹点 [x, y, z]
        """
        N = traj_ego.shape[0]
        traj_lidar = np.zeros((N, 3), dtype=np.float32)

        # 转换位置 (x, y, z=0)
        for i in range(N):
            # 构建齐次坐标（z 设为 z_height，通常是地面高度 0）
            ego_pos = np.array([traj_ego[i, 0], traj_ego[i, 1], z_height, 1.0])

            # 应用 ego2lidar 转换
            lidar_pos = self.config.ego2lidar @ ego_pos

            traj_lidar[i, 0] = lidar_pos[0]
            traj_lidar[i, 1] = lidar_pos[1]
            traj_lidar[i, 2] = lidar_pos[2]

        return traj_lidar

    def _euler_to_rotation_matrix(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        """
        欧拉角（ZYX顺序）转换为旋转矩阵

        Args:
            roll_deg: roll 角度（绕 X 轴旋转）
            pitch_deg: pitch 角度（绕 Y 轴旋转）
            yaw_deg: yaw 角度（绕 Z 轴旋转）

        Returns:
            R: (3, 3) 旋转矩阵
        """
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        # Rz (yaw)
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0,              0,              1]
        ])

        # Ry (pitch)
        Ry = np.array([
            [math.cos(pitch),  0, -math.sin(pitch)],
            [0,                1, 0              ],
            [math.sin(pitch),  0,  math.cos(pitch)]
        ])

        # Rx (roll)
        Rx = np.array([
            [1, 0,               0              ],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll),  math.cos(roll)]
        ])

        # ZYX 顺序: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    def _build_rslidar2camera_matrix(self) -> np.ndarray:
        """
        构建 rslidar → camera 外参转换矩阵

        外参描述的是 camera 在 lidar 坐标系中的位置/姿态，投影需要取逆
        同时应用坐标轴转换：LiDAR(X前Y左Z上) → Camera(Z前X右Y下)

        Returns:
            RT: (4, 4) 齐次变换矩阵 (lidar → camera)
        """
        # 坐标系轴转换矩阵
        R_axis = self.config.lidar_to_camera_axis_transform

        # 安装偏差旋转矩阵
        euler_deg = self.config.camera_to_rslidar_euler_deg
        R_install = self._euler_to_rotation_matrix(euler_deg[0], euler_deg[1], euler_deg[2])

        # 平移向量（相机在 lidar 坐标系中的位置）
        t_install = self.config.camera_to_rslidar_translation

        # 构建 camera → lidar 矩阵，取逆得到 lidar → camera
        T_cam_to_lidar = np.eye(4)
        T_cam_to_lidar[:3, :3] = R_install
        T_cam_to_lidar[:3, 3] = t_install
        T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)

        # 组合轴转换
        T_axis = np.eye(4)
        T_axis[:3, :3] = R_axis
        RT = T_axis @ T_lidar_to_cam

        return RT

    def _project_points_to_camera(self, points_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 rslidar frame 的 3D 点投影到相机图像平面

        Args:
            points_lidar: (N, 3) rslidar frame 中的点 [x, y, z]

        Returns:
            points_2d: (N, 2) 投影后的像素坐标 [u, v]
            valid_mask: (N,) 有效点掩码（在图像范围内的点）
        """
        N = points_lidar.shape[0]
        if N == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=bool)

        # 构建 RT 矩阵
        RT = self._build_rslidar2camera_matrix()

        # 转换到齐次坐标
        points_homo = np.hstack([points_lidar, np.ones((N, 1))])  # (N, 4)

        # 转换到相机坐标系
        points_camera = (RT @ points_homo.T).T[:, :3]  # (N, 3)

        # 使用 OpenCV 进行投影（含畸变校正）
        K = self.config.single_camera_intrinsic
        D = self.config.single_camera_distortion

        # cv2.projectPoints 需要 (N, 1, 3) 格式
        points_camera_cv = points_camera.reshape(-1, 1, 3).astype(np.float64)

        points_2d, _ = cv2.projectPoints(
            points_camera_cv,
            np.zeros(3, dtype=np.float64),  # 无额外旋转
            np.zeros(3, dtype=np.float64),  # 无额外平移
            K.astype(np.float64),
            D.astype(np.float64)
        )

        points_2d = points_2d.reshape(-1, 2)  # (N, 2)

        # 过滤超出图像范围的点
        u_valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.config.image_width)
        v_valid = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.config.image_height)
        valid_mask = u_valid & v_valid

        return points_2d, valid_mask

    def _draw_trajectory_on_image(self, image: np.ndarray, points_2d: np.ndarray,
                                   valid_mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
                                   line_thickness: int = 2, point_radius: int = 5) -> np.ndarray:
        """
        在图像上绘制轨迹（线 + 点标记）+ 自车参考点

        Args:
            image: (H, W, 3) numpy 图像（RGB）
            points_2d: (N, 2) 投影后的像素坐标 [u, v]
            valid_mask: (N,) 有效点掩码
            color: RGB 颜色，默认绿色
            line_thickness: 线宽，默认 2
            point_radius: 点半径，默认 5

        Returns:
            annotated_image: 标注后的图像
        """
        annotated = image.copy()
        H, W = image.shape[:2]

        # 自车参考点（图像底部中心偏左，底部偏上一点避免超出边界）
        ego_point = (W // 2 - 50, H - 5)
        ego_radius = 10

        # 获取有效轨迹点
        valid_points = points_2d[valid_mask]
        if len(valid_points) < 1:
            # 仅绘制自车点
            cv2.circle(annotated, ego_point, ego_radius, color, -1)
            return annotated

        valid_points_int = valid_points.astype(np.int32)

        # 找到距离自车参考点最近的轨迹点
        distances = np.sqrt((valid_points_int[:, 0] - ego_point[0])**2 +
                           (valid_points_int[:, 1] - ego_point[1])**2)
        nearest_idx = np.argmin(distances)
        nearest_point = tuple(valid_points_int[nearest_idx])

        # 绘制自车点到最近轨迹点的连线（统一颜色和线宽）
        cv2.line(annotated, ego_point, nearest_point, color, line_thickness)

        # 绘制轨迹线（连接相邻点）
        for i in range(len(valid_points_int) - 1):
            pt1 = tuple(valid_points_int[i])
            pt2 = tuple(valid_points_int[i + 1])
            cv2.line(annotated, pt1, pt2, color, line_thickness)

        # 绘制自车点（较大）
        cv2.circle(annotated, ego_point, ego_radius, color, -1)

        # 绘制轨迹点标记
        for pt in valid_points_int:
            cv2.circle(annotated, tuple(pt), point_radius, color, -1)

        return annotated

    def _publish_fpv_trajectory_view(self, image_single_np: np.ndarray, traj_ego: np.ndarray):
        """
        发布第一视角轨迹可视化图像

        Args:
            image_single_np: (1080, 1920, 3) 单相机原始图像
            traj_ego: (N, 3) 预测轨迹 (ego frame) [x, y, heading]
        """
        try:
            # 1. 将预测轨迹从 ego frame 转换到 rslidar frame
            traj_lidar = self._transform_to_lidar_frame(traj_ego)

            # 2. 投影到相机图像平面（使用内参和外参）
            points_2d, valid_mask = self._project_points_to_camera(traj_lidar)

            # 3. 裁剪图像：顶部截 200，底部截 100
            crop_top = 200
            crop_bottom = 100
            cropped_image = image_single_np[crop_top:-crop_bottom, :]  # (780, 1920, 3)

            # 4. 调整投影点 y 坐标（减去裁剪起始行）
            points_2d_adjusted = points_2d.copy()
            points_2d_adjusted[:, 1] -= crop_top

            # 5. 更新有效掩码（检查裁剪后的图像范围）
            cropped_height = image_single_np.shape[0] - crop_top - crop_bottom
            valid_mask_final = (points_2d_adjusted[:, 0] >= 0) & (points_2d_adjusted[:, 0] < self.config.image_width)
            valid_mask_final &= (points_2d_adjusted[:, 1] >= 0) & (points_2d_adjusted[:, 1] < cropped_height)

            # 6. 在裁剪后的图像上绘制轨迹
            annotated_image = self._draw_trajectory_on_image(
                cropped_image, points_2d_adjusted, valid_mask_final,
                color=(0, 255, 0),      # 绿色
                line_thickness=10,      # 加粗线宽
                point_radius=12         # 加粗点大小
            )

            # 7. 发布
            ros_image = self._numpy_to_ros_image(annotated_image, encoding='rgb8', frame_id='camera')
            self.fpv_trajectory_pub.publish(ros_image)

        except Exception as e:
            rospy.logwarn(f"第一视角轨迹可视化发布失败：{e}")

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def _publish_input_image(self, input_image: np.ndarray):
        """发布输入图像（拼接/裁剪后）"""
        try:
            ros_image = self._numpy_to_ros_image(input_image, encoding='rgb8', frame_id='camera_front')
            self.input_image_pub.publish(ros_image)
        except Exception as e:
            rospy.logwarn(f"发布输入图像失败：{e}")

    def _publish_history_marker(self, local_history: np.ndarray):
        """
        发布本地历史轨迹为蓝色 LINE_STRIP Marker（在 rslidar frame 中）

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

        # 构建 Marker 消息
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "history_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # 蓝色：r=0, g=0, b=1, a=1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # 线宽 0.1m
        marker.scale.x = 0.15

        # 添加轨迹点
        for pt in local_history:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = float(pt[2]) if len(pt) > 2 else 0.0
            marker.points.append(point)

        self.history_marker_pub.publish(marker)


    def _publish_prediction_marker(self, traj_ego: np.ndarray):
        """
        发布预测轨迹为绿色 LINE_STRIP Marker（在 rslidar frame 中）

        Args:
            traj_ego: (N, 3) numpy array, 每行为 [x, y, yaw] 在 ego frame 中
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

        # 构建 Marker 消息
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "prediction_trajectory"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # 绿色：r=0, g=1, b=0, a=1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # 线宽 0.15m（稍粗以区分）
        marker.scale.x = 0.15

        # 添加轨迹点
        for pt in traj_ego:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = float(pt[2]) if len(pt) > 2 else 0.0
            marker.points.append(point)

        self.prediction_marker_pub.publish(marker)
        

    def _infer(self, command_one_hot: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        raw_lidar = self._parse_pointcloud2(self.buffer.latest_pointcloud)

        # 始终解码单相机图像（用于第一视角轨迹可视化）
        image_single_np = self._decode_image(self.buffer.latest_image_single)

        # 模型输入根据 image_input_mode 决定
        if self.config.image_input_mode == "dual_camera":
            print("[INFO] Camera input mode: dual_camera")
            image_left_np = self._decode_image(self.buffer.latest_image_left)
            image_right_np = self._decode_image(self.buffer.latest_image_right)
            camera_feature, vis_image = self._build_camera_feature(image_left_np, image_right_np)
        else:
            print("[INFO] Camera input mode: single_camera")
            camera_feature, vis_image = self._build_camera_feature_single(image_single_np)

        t1 = time.perf_counter()

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

        # 发布输入图像（可视化）
        self._publish_input_image(vis_image)

        # 第一视角轨迹可视化（始终使用单相机图像）
        self._publish_fpv_trajectory_view(image_single_np, pred_traj_np)

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
                    command_one_hot = self.command_to_onehot(carla_command=self.current_command, expand=False)
                    traj_ego = self._infer(command_one_hot)
                    local_history = self.buffer.get_local_history_trajectory()
                    self._publish_history_marker(local_history)
                    self._publish_prediction_marker(traj_ego)
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