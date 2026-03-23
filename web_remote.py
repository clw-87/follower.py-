# 透過 Flask 架設區域網頁，利用手機陀螺儀與按鈕遠端遙控車輛。
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from flask import Flask, render_template_string, request
import threading
import time

# --- 1. 設定網頁伺服器 ---
app = Flask(__name__)

# 這是體感控制的網頁 (包含 JavaScript 讀取手機陀螺儀)
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { font-family: sans-serif; text-align: center; background-color: #222; color: white; user-select: none; }
        #status { font-size: 18px; color: #aaa; margin: 20px; }
        .btn-start { 
            background-color: #4CAF50; color: white; width: 200px; height: 80px; 
            font-size: 24px; border-radius: 15px; border: none; margin-top: 50px;
        }
        .data-box { border: 1px solid #555; padding: 10px; margin: 10px; border-radius: 10px; }
    </style>
</head>
<body>
    <h1>🏎️ 體感遙控模式</h1>
    
    <div id="control-area" style="display:none;">
        <div class="data-box">
            <p>前後傾斜 (油門): <span id="beta-val">0</span></p>
            <p>左右傾斜 (方向): <span id="gamma-val">0</span></p>
        </div>
        <p style="color: yellow;">💡 請將手機橫放操作</p>
        <button onclick="stopCar()" style="background-color:red; width:100px; height:50px; border:none; border-radius:10px; color:white;">緊急煞車</button>
    </div>

    <button id="start-btn" class="btn-start" onclick="requestPermission()">點我啟動引擎</button>
    <p id="status">等待啟動...</p>

    <script>
        let last_sent = 0;
        
        function requestPermission() {
            // iOS 13+ 需要使用者同意才能存取陀螺儀
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                DeviceOrientationEvent.requestPermission()
                    .then(permissionState => {
                        if (permissionState === 'granted') {
                            startControl();
                        } else {
                            alert("需要權限才能開車喔！");
                        }
                    })
                    .catch(console.error);
            } else {
                // Android 或舊版 iOS 直接啟動
                startControl();
            }
        }

        function startControl() {
            document.getElementById('start-btn').style.display = 'none';
            document.getElementById('control-area').style.display = 'block';
            document.getElementById('status').innerText = "感應器運作中...";
            
            window.addEventListener('deviceorientation', handleOrientation);
        }

        function handleOrientation(event) {
            let now = Date.now();
            if (now - last_sent < 100) return; // 限制發送頻率 (每 100ms 一次)
            last_sent = now;

            // beta: 前後傾斜 (-180 到 180), gamma: 左右傾斜 (-90 到 90)
            // 注意：不同手機方向定義可能略有不同，這裡以橫拿手機為主
            let x = event.beta;  // 前後
            let y = event.gamma; // 左右

            // 顯示在螢幕上
            document.getElementById('beta-val').innerText = Math.round(x);
            document.getElementById('gamma-val').innerText = Math.round(y);

            // 傳送數據給 Python
            fetch('/update_sensor', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({beta: x, gamma: y})
            });
        }
        
        function stopCar() {
            fetch('/update_sensor', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({beta: 0, gamma: 0, stop: true})
            });
        }
    </script>
</body>
</html>
"""

# --- 2. 設定 ROS2 節點 ---
class GyroTeleop(Node):
    def __init__(self):
        super().__init__('gyro_teleop')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('體感遙控器已啟動！')

    def update_speed(self, beta, gamma):
        msg = Twist()
        
        # --- 靈敏度調整區 ---
        # beta (前後): 通常手機平放是 0，前傾是正，後傾是負 (或相反，視手機而定)
        # gamma (左右): 左傾負，右傾正
        
        # 設定死區 (Deadzone)，避免手抖車子亂動
        if abs(beta) < 5: beta = 0
        if abs(gamma) < 5: gamma = 0

        # 映射數值：假設傾斜 45 度達到最高速
        max_speed = 0.22  # TurtleBot3 Burger 最高速
        max_turn = 1.0
        
        # 簡單的線性轉換
        linear_x = (beta / 45.0) * max_speed
        angular_z = (gamma / 45.0) * max_turn * -1 # 乘 -1 是為了反轉方向(如果不順手可以拿掉)

        # 限制最大值
        linear_x = max(min(linear_x, max_speed), -max_speed)
        angular_z = max(min(angular_z, max_turn), -max_turn)

        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        
        self.publisher_.publish(msg)

# 全域變數
ros_node = None

# --- 3. 網頁路由 ---
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/update_sensor', methods=['POST'])
def update_sensor():
    data = request.json
    
    if 'stop' in data:
        ros_node.update_speed(0, 0)
        return "Stopped"

    # 讀取手機傳來的角度
    beta = data.get('beta', 0)   # 前後
    gamma = data.get('gamma', 0) # 左右
    
    # 呼叫 ROS2 發布速度
    if ros_node:
        ros_node.update_speed(beta, gamma)
        
    return "OK"

def run_ros_node():
    rclpy.spin(ros_node)

def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = GyroTeleop()
    
    threading.Thread(target=run_ros_node, daemon=True).start()
    
    # 啟動伺服器
    app.run(host='0.0.0.0', port=5000)

    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
