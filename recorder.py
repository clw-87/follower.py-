# 記錄遙控駕駛時的 Odom (里程計) X, Y 座標，並匯出成 CSV 檔案。
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import math
import os

class TrackRecorder(Node):
    def __init__(self):
        super().__init__('track_recorder')

        # 訂閱里程計 (Odom) 資訊
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # 設定存檔路徑 (存放在使用者的家目錄下)
        self.file_path = os.path.expanduser('~/race_track_data.csv')

        # 初始化變數
        self.last_x = 0.0
        self.last_y = 0.0
        self.record_threshold = 0.1  # 靈敏度：每移動 0.1 公尺 (10公分) 記錄一次

        # 程式啟動時，先建立 CSV 檔案並寫入標頭
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y']) 

        self.get_logger().info(f'🔴 開始錄製！路徑檔將存於: {self.file_path}')

    def odom_callback(self, msg):
        # 從訊息中提取位置
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 計算距離上一個記錄點有多遠
        distance = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)

        # 如果移動距離超過設定門檻，就記錄下來
        if distance > self.record_threshold:
            self.save_to_csv(x, y)
            self.last_x = x
            self.last_y = y
            self.get_logger().info(f'✅ 記錄點: x={x:.2f}, y={y:.2f}')

    def save_to_csv(self, x, y):
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([x, y])

def main(args=None):
    rclpy.init(args=args)
    node = TrackRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
