# follower.py-
# 讀取 CSV 座標檔，使用純追蹤 (Pure Pursuit) 演算法讓車輛盲跑。
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import csv
import math
import os

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        
        # 訂閱位置資訊
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # 發布速度指令
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 讀取路徑檔
        self.path = []
        self.file_path = os.path.expanduser('~/race_track_data.csv')
        self.load_path()
        
        self.current_idx = 0  # 目前追蹤到第幾個點
        self.target_speed = 0.15 # 直線速度 (m/s) - 建議先設慢一點
        self.nav_threshold = 0.2 # 距離目標多近算到達 (公尺)
        
        self.get_logger().info('🚗 自動駕駛模組已啟動！等待里程計訊號...')

    def load_path(self):
        try:
            with open(self.file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader) # 跳過標題
                for row in reader:
                    # 讀取 x, y 座標 (轉成 float)
                    self.path.append((float(row[0]), float(row[1])))
            self.get_logger().info(f'成功讀取賽道，共有 {len(self.path)} 個路徑點')
        except Exception as e:
            self.get_logger().error(f'讀取檔案失敗: {e}')

    def odom_callback(self, msg):
        if not self.path:
            return

        # 1. 取得車子當前位置與方向
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        yaw = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

        # 2. 取得當前目標點
        target_x, target_y = self.path[self.current_idx]

        # 3. 計算與目標的距離
        dist = math.sqrt((target_x - curr_x)**2 + (target_y - curr_y)**2)

        # 4. 如果夠近了，就換下一個點
        if dist < self.nav_threshold:
            self.current_idx += 1
            if self.current_idx >= len(self.path):
                self.current_idx = 0 # 跑完一圈，從頭開始循環
                self.get_logger().info('🏁 跑完一圈！重新開始...')
            return # 下次迴圈再處理新目標

        # 5. 計算導航角度 (PID 控制的核心)
        # 目標角度 = atan2(dy, dx)
        desired_yaw = math.atan2(target_y - curr_y, target_x - curr_x)
        
        # 角度誤差 = 目標角度 - 當前車頭角度
        yaw_error = desired_yaw - yaw

        # 正規化角度誤差 (限制在 -PI 到 +PI 之間)
        while yaw_error > math.pi: yaw_error -= 2 * math.pi
        while yaw_error < -math.pi: yaw_error += 2 * math.pi

        # 6. 發送控制指令
        cmd = Twist()
        cmd.linear.x = self.target_speed
        cmd.angular.z = 1.5 * yaw_error  # P 控制器 (1.5 是轉向靈敏度，可調整)

        self.publisher_.publish(cmd)
        
        # 顯示除錯資訊 (每10次顯示一次以免洗版)
        # self.get_logger().info(f'追蹤點 {self.current_idx}: 距離 {dist:.2f}m, 誤差 {yaw_error:.2f}')

    def get_yaw_from_quaternion(self, q):
        # 將四元數轉換為偏航角 (Yaw)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    follower = PathFollower()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        # 結束時停車
        follower.publisher_.publish(Twist()) 
    finally:
        follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
