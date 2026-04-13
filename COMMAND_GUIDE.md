# DiffusionDrive 导航命令控制指南

## 命令映射

| 命令值 | 含义 | one-hot 编码 |
|--------|------|--------------|
| `1` | 左转 (LEFT) | `[1, 0, 0, 0]` |
| `2` | 右转 (RIGHT) | `[0, 0, 1, 0]` |
| `3` | 直行 (STRAIGHT) | `[0, 1, 0, 0]` |

> 默认命令为 `3` (直行)

## Topic 信息

- **Topic**: `/diffusiondrive/command`
- **消息类型**: `std_msgs/Int32`

## 终端使用示例

### 单次发送

```bash
# 直行
rostopic pub /diffusiondrive/command std_msgs/Int32 "data: 3" --once

# 左转
rostopic pub /diffusiondrive/command std_msgs/Int32 "data: 1" --once

# 右转
rostopic pub /diffusiondrive/command std_msgs/Int32 "data: 2" --once
```

### 持续发送 (按频率)

```bash
# 每秒发送一次左转命令
rostopic pub /diffusiondrive/command std_msgs/Int32 "data: 1" -r 1

# 每0.5秒发送一次直行命令
rostopic pub /diffusiondrive/command std_msgs/Int32 "data: 3" -r 2
```

### 查看当前命令

```bash
# 监听命令 topic
rostopic echo /diffusiondrive/command

# 查看 topic 信息
rostopic info /diffusiondrive/command
```

## Python 调用示例

```python
import rospy
from std_msgs.msg import Int32

rospy.init_node('command_sender')
pub = rospy.Publisher('/diffusiondrive/command', Int32, queue_size=10)

# 发送左转命令
pub.publish(Int32(data=1))

# 发送直行命令
pub.publish(Int32(data=3))
```

## RViz 可视化

轨迹 Marker 颜色：
- **历史轨迹**: 蓝色 (`/diffusiondrive/history_marker`)
- **预测轨迹**: 绿色 (`/diffusiondrive/prediction_marker`)

## 注意事项

1. 无效命令值会被忽略并回退到默认直行 (`3`)
2. 命令更新时会打印日志：`导航命令更新: X (左转/右转/直行)`
3. 命令在推理循环中实时生效，无需重启节点