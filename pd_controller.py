"""
平滑 PID 控制器 - 用于机器人轨迹跟踪

Control law:
    theta = atan2(dy, dx)  # heading error
    w = sat(KP * filtered_theta + KD * d_theta/dt)
    v = sat(KP * filtered_distance + KD * d_distance/dt) * cos(filtered_theta)

Features:
    - EMA filter on both theta and distance to reduce noise
    - PD control for both v and w
    - Deadband to suppress tiny steering corrections
    - Slew-rate limit for smooth velocity transition
    - Velocity reduction when turning (cos coupling)
"""

import numpy as np
from typing import Tuple, Optional
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

EPSILON = 1e-8  # 数值下界


def wrap_to_pi(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return float(theta)


class PIDController:
    """平滑 PID 控制器 - 用于机器人轨迹跟踪"""

    def __init__(self, params: Optional[dict] = None):
        """
        初始化 PID 控制器

        Args:
            params: 参数字典，使用默认值时可传 None
        """
        # 默认参数
        default_params = {
            # 机器人限制参数 (LIMIT_)
            'LIMIT_MAX_LINEAR_VELOCITY': 1.4,    # m/s - 最大线速度
            'LIMIT_MAX_ANGULAR_VELOCITY': 0.5,   # rad/s - 最大角速度
            'LIMIT_MAX_LINEAR_ACCEL': 2.0,       # m/s² - 最大线加速度
            'LIMIT_MAX_ANGULAR_ACCEL': 1.0,      # rad/s² - 最大角加速度

            # 转向控制参数 (STEER_)
            'STEER_KP': 1.4,                     # 比例增益 - 转向响应强度
            'STEER_KD': 0.2,                     # 微分增益 - 抑制振荡
            'STEER_DEADBAND': 0.05,              # rad (~1.7°) - 死区阈值

            # 平滑滤波参数 (SMOOTH_)
            'SMOOTH_EMA_ALPHA_THETA': 0.25,      # theta EMA滤波系数
            'SMOOTH_EMA_ALPHA_DISTANCE': 0.15,   # distance EMA滤波系数

            # 速度控制参数 (VEL_)
            'VEL_KP': 0.4,                       # 比例增益 - 距离→速度响应强度
            'VEL_KD': 0.2,                       # 微分增益 - 距离变化率
        }

        # 合并用户参数
        self.params = {**default_params, **(params or {})}

        # 内部状态变量
        self.filtered_theta = 0.0       # EMA滤波后的航向角误差
        self.filtered_distance = 0.0    # EMA滤波后的距离
        self.theta_last = 0.0           # 上时刻航向角误差
        self.distance_last = 0.0        # 上时刻距离
        self.v_last = 0.0               # 上时刻线速度
        self.w_last = 0.0               # 上时刻角速度

    def step(self, wpt: np.ndarray, dt: float) -> Tuple[float, float]:
        """
        执行单步控制计算

        Args:
            wpt: waypoint array, [dx, dy] 或 [dx, dy, hx, hy]
            dt: 时间步长 (秒)

        Returns:
            (v, w): 线速度 (m/s), 角速度 (rad/s)
        """
        # 提取参数
        p = self.params
        STEER_KP = p['STEER_KP']
        STEER_KD = p['STEER_KD']
        STEER_DEADBAND = p['STEER_DEADBAND']
        SMOOTH_EMA_ALPHA_THETA = p['SMOOTH_EMA_ALPHA_THETA']
        SMOOTH_EMA_ALPHA_DISTANCE = p['SMOOTH_EMA_ALPHA_DISTANCE']
        VEL_KP = p['VEL_KP']
        VEL_KD = p['VEL_KD']
        LIMIT_MAX_LINEAR_VELOCITY = p['LIMIT_MAX_LINEAR_VELOCITY']
        LIMIT_MAX_ANGULAR_VELOCITY = p['LIMIT_MAX_ANGULAR_VELOCITY']
        LIMIT_MAX_LINEAR_ACCEL = p['LIMIT_MAX_LINEAR_ACCEL']
        LIMIT_MAX_ANGULAR_ACCEL = p['LIMIT_MAX_ANGULAR_ACCEL']

        # 解析 waypoint
        dx, dy = float(wpt[0]), float(wpt[1])
        distance = float(np.hypot(dx, dy))

        # 计算航向角误差 theta = atan2(dy, dx)
        if len(wpt) == 4 and distance < 0.05:  # near goal: use predicted heading
            hx, hy = float(wpt[2]), float(wpt[3])
            if abs(hx) > EPSILON or abs(hy) > EPSILON:
                theta = wrap_to_pi(np.arctan2(hy, hx))
            else:
                theta = 0.0
        else:
            theta = wrap_to_pi(np.arctan2(dy, dx))

        dt = max(float(dt), 1e-3)

        # === 角速度控制 ===
        # EMA 低通滤波
        self.filtered_theta = (1.0 - SMOOTH_EMA_ALPHA_THETA) * self.filtered_theta + SMOOTH_EMA_ALPHA_THETA * theta

        # 微分项
        d_theta = wrap_to_pi(self.filtered_theta - self.theta_last) / dt
        self.theta_last = self.filtered_theta

        # PD 控制律
        w = STEER_KP * self.filtered_theta + STEER_KD * d_theta

        # 死区
        if abs(self.filtered_theta) < STEER_DEADBAND:
            w = 0.0

        # 限幅
        w = float(np.clip(w, -LIMIT_MAX_ANGULAR_VELOCITY, LIMIT_MAX_ANGULAR_VELOCITY))

        # Slew-rate limit
        w = float(np.clip(w, self.w_last - LIMIT_MAX_ANGULAR_ACCEL * dt, self.w_last + LIMIT_MAX_ANGULAR_ACCEL * dt))
        self.w_last = w

        # === 线速度控制 ===
        # EMA 低通滤波
        self.filtered_distance = (1.0 - SMOOTH_EMA_ALPHA_DISTANCE) * self.filtered_distance + SMOOTH_EMA_ALPHA_DISTANCE * distance

        # 微分项
        d_distance = (self.filtered_distance - self.distance_last) / dt
        self.distance_last = self.filtered_distance

        # PD 控制律 + cos coupling
        v = (VEL_KP * self.filtered_distance + VEL_KD * d_distance) * max(0.0, float(np.cos(self.filtered_theta)))
        v = float(np.clip(v, 0.0, LIMIT_MAX_LINEAR_VELOCITY))

        # Slew-rate limit
        v = float(np.clip(v, self.v_last - LIMIT_MAX_LINEAR_ACCEL * dt, self.v_last + LIMIT_MAX_LINEAR_ACCEL * dt))
        self.v_last = v

        return v, w

    def reset(self):
        """重置内部状态变量"""
        self.filtered_theta = 0.0
        self.filtered_distance = 0.0
        self.theta_last = 0.0
        self.distance_last = 0.0
        self.v_last = 0.0
        self.w_last = 0.0

    def load_params_from_ros(self):
        """从 ROS param server 加载参数"""
        for key in self.params:
            try:
                self.params[key] = rospy.get_param(f'~{key}', self.params[key])
            except rospy.ROSException:
                pass  # 使用默认值


# ==============================================================================
# ROS 节点入口
# ==============================================================================
VEL_TOPIC = "/cmd_vel"
WAYPOINT_TOPIC = "/diffusiondrive/waypoint"
RATE = 10  # Hz - 控制频率


def main():
    rospy.init_node("PD_CONTROLLER_SMOOTH", anonymous=False)

    # 初始化控制器
    controller = PIDController()
    controller.load_params_from_ros()

    waypoint = None

    def callback_drive(waypoint_msg: Float32MultiArray):
        nonlocal waypoint
        waypoint = waypoint_msg.data

    rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
    vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)

    rate = rospy.Rate(RATE)
    rospy.loginfo("Registered with master node. Waiting for waypoints...")

    last_t = rospy.Time.now()

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        dt = (now - last_t).to_sec()
        last_t = now

        vel_msg = Twist()
        if waypoint is not None:
            wpt = np.array(waypoint, dtype=np.float64)
            v, w = controller.step(wpt, dt)
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            rospy.loginfo_throttle(0.2, f"cmd_vel: v={v:.3f}, w={w:.3f}, dt={dt:.3f}")
        vel_out.publish(vel_msg)
        rate.sleep()


if __name__ == "__main__":
    main()