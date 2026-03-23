# 僅使用 OpenCV 影像處理，過濾白線或黃線並計算重心，控制車輛轉向。
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
#撰寫視覺巡線程式
class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        
        # 1. 訂閱影像話題 (來自 v4l2_camera)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        
        # 2. 發布速度控制
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 3. 建立 OpenCV 轉換器
        self.bridge = CvBridge()
        
        self.get_logger().info('👁️ 視覺巡線模式啟動！尋找白色跑道線...')

    def image_callback(self, msg):
        # ... (前面的轉換與裁切程式碼同上) ...
        
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # 1. 定義白色的範圍
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # 2. 定義黃色的範圍
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 3. 合併兩個遮罩 (只要是白 OR 黃都算)
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        M = cv2.moments(mask)
        
        # ... (後面的 PID 控制邏輯保持不變) ...
        
        cmd = Twist()
        
        if M['m00'] > 0:
            # 找到了線！計算中心點 cx
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # 畫面中心點
            img_center = width / 2
            
            # 誤差 = 線的中心 - 畫面中心
            error = cx - img_center
            
            # --- P 控制器 (修正方向) ---
            # 負值代表線在左邊 -> 要左轉 (Angular +)
            # 正值代表線在右邊 -> 要右轉 (Angular -)
            # 因為 ROS 座標系左轉是正，所以我們要取反號
            kp = 0.005 # 轉向靈敏度 (如果太晃改小，轉不過去改大)
            
            cmd.angular.z = -float(error) * kp
            cmd.linear.x = 0.15 # 看到線就前進
            
            # self.get_logger().info(f'誤差: {error} | 轉向: {cmd.angular.z:.2f}')
            
        else:
            # 沒看到線 (可能跑出去了，或是光線太暗)
            self.get_logger().warning('❌ 看不到路！停車搜尋中...')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3 # 原地慢速旋轉找路
            
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publisher_.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
