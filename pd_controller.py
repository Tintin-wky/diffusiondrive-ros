import numpy as np
from typing import Tuple
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

vel_msg = Twist()  # 初始化为零速度
MAX_V = 1.5  # m/s
MAX_W = 0.5  # rad/s
VEL_TOPIC = "/cmd_vel"
DT = 0.1
RATE = 10
EPS = 1e-8


def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      

def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
	"""PD controller for the robot"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/DT		
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*DT)
	else:
		v = dx / DT
		w = np.arctan(dy/dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w

def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	waypoint = waypoint_msg.data
	v, w = pd_controller(waypoint)
	vel_msg.linear.x = v
	vel_msg.angular.z = w
	waypoint_str = ", ".join([f"{x:.2f}" for x in waypoint])
	rospy.loginfo("Waypoint: [%s] -> cmd_vel: v=%.2f, w=%.2f", waypoint_str, v, w)

def main():
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint_sub = rospy.Subscriber("/diffusiondrive/waypoint", Float32MultiArray, callback_drive, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	rospy.loginfo("Registered with master node. Waiting for waypoints...")
	while not rospy.is_shutdown():
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
