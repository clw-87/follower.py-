import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import threading
import random
from flask import Flask, Response
from rclpy.qos import qos_profile_sensor_data

# =============================================================================
# 常數定義（針對此跑道調整）
# =============================================================================
EMERGENCY_STOP_DIST   = 0.10   # 緊急停止距離 (m)
OBSTACLE_STOP_DIST    = 0.15   # 障礙物停止距離，稍微放寬給反應時間 (m)
LIDAR_SAFE_DIST       = 0.35   # 避障安全距離 (m)，跑道窄故設小一點
ARUCO_DETECT_INTERVAL = 0.3    # ArUco 偵測間隔 (s)，加快辨識頻率
ACTION_DELAY_TIME     = 0.8    # 標記辨識後延遲觸發時間 (s)，縮短讓反應更即時
CORRIDOR_WIDTH        = 220    # 虛擬車道寬度 (px)，配合跑道寬度調小
FRAME_WIDTH           = 320
FRAME_HEIGHT          = 240

# =============================================================================
# 跑道專屬調參區（可直接在這裡微調，不需動邏輯）
# =============================================================================

# 三連彎各階段時間 (秒) [0不用, 1=右, 2=左, 3=右]
# 從跑道圖判斷：第一右彎短、中間左彎最長、最後右彎中等
S_BEND_DURATIONS = [0.0, 1.2, 2.0, 1.3]
S_BEND_SPEED     = 0.04   # 三連彎速度，極慢避免出界

# 第一個彎：60度左轉（ID 1）— 角度較小，轉向力與時間都縮短
TURN_90_SPEED    = 0.05
TURN_90_STEER    = -0.85  # 60度：比90度力道小（90度約-1.3）
TURN_90_DURATION = 1.3    # 60度：時間比90度短

# 第二個彎：90度右轉（ID 6）
TURN_120_SPEED      = 0.05
TURN_120_STEER      = -1.3   # 90度右轉力道
TURN_120_DURATION   = 2.0    # 90度持續時間
TURN_120_MASK_RATIO = 0.5    # 遮蔽左側50%

# 避障區（根據跑道：障礙物偏左，預設往右繞）
AVOIDING_ENTRY_SPEED   = 0.10   # 進入避障區後的緩衝減速目標速度
AVOIDING_ENTRY_TIME    = 1.5    # 緩衝減速持續時間 (s)，此段先減速再開始閃避
AVOIDING_SPEED         = 0.04   # 避障全程慢速（減速完成後鎖定此速度）
AVOIDING_DURATION      = 22.0   # 整段避障區最長時間 (s)
AVOIDING_REPULSION     = 6.0    # 雷達排斥力道
AVOIDING_TURN_BIAS     = 0.55   # 雷達佔融合比例（提高，讓雷達更主導）
AVOIDING_DEFAULT_BIAS  = -0.25  # 預設右偏偏置（負=右），無障礙物時仍略偏右通過
AVOIDING_SAFE_DIST     = 0.40   # 開始排斥的距離 (m)
AVOIDING_DANGER_DIST   = 0.18   # 危險距離，強制閃避 (m)，調小讓觸發更靈敏

# 直線衝刺
STRAIGHT_SPEED = 0.22

# 停止
STOP_DURATION = -1.0  # -1 = 永久停止，不自動解除

# 轉彎後補償小轉（POST_TURN）
# 每個轉彎動作結束後，自動接一小段同向補償，讓車頭對齊新車道
POST_TURN_SPEED    = 0.04  # 補償時速度極慢
POST_TURN_STEER    = 0.5   # 補償轉向力（絕對值），方向由主彎決定
POST_TURN_DURATION = 0.5   # 補償持續時間 (秒)，可視情況微調

# =============================================================================
# Flask 網頁伺服器
# =============================================================================
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

@app.route('/')
def index():
    return (
        "<html><body style='background-color:#111; color:white; text-align:center; font-family:sans-serif;'>"
        "<h1 style='color:#FFD700;'>🏎️ 自駕車即時監控</h1>"
        "<img src='/video_feed' style='width:100%; max-width:800px; border:3px solid #FFD700; border-radius:8px;'>"
        "</body></html>"
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    global output_frame
    while True:
        with lock:
            frame = output_frame.copy() if output_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        flag, encodedImage = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not flag:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + bytearray(encodedImage)
            + b'\r\n'
        )
        time.sleep(0.033)


# =============================================================================
# ROS2 主節點
# =============================================================================
class FusionDriver(Node):
    def __init__(self):
        super().__init__('fusion_driver')

        self.img_sub  = self.create_subscription(Image,     '/image_raw', self.image_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, '/scan',      self.scan_callback,  qos_profile_sensor_data)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.bridge = CvBridge()

        # 雷達距離
        self.front_dist = 10.0
        self.dist_left  = 10.0
        self.dist_right = 10.0
        # 左前/右前方位（避障用）
        self.dist_front_left  = 10.0
        self.dist_front_right = 10.0

        # 車道線狀態
        self.line_error    = 0.0
        self.line_detected = False
        self.system_state  = "CRUISING"

        # 速度/轉向
        self.current_speed     = 0.0
        self.current_steer     = 0.0
        self.prev_error        = 0.0
        self.integral_error    = 0.0   # 新增積分項，改善穩態誤差
        self.deep_steer_memory = 0.0

        # 緊急停止旗標（只在此設旗，由 control_loop publish）
        self._emergency_stop_flag = False

        # 動作系統
        self.planned_action          = "NONE"
        self.plan_time               = 0.0
        self.current_action          = "NONE"
        self.action_start_time       = 0.0
        self.current_action_duration = 0.0
        self.race_finished           = False  # 終點旗標，永久停止用

        self.delay_time = ACTION_DELAY_TIME

        # POST_TURN 補償小轉狀態
        self.post_turn_active    = False   # 是否正在補償
        self.post_turn_start     = 0.0     # 補償開始時間
        self.post_turn_direction = 1.0     # +1=左補償, -1=右補償

        # 三連彎
        self.s_bend_stage_durations = S_BEND_DURATIONS
        self.s_bend_current_stage   = 0
        self.s_bend_stage_start     = 0.0  # 每個 stage 自己的計時基準

        self.last_seen_time  = time.time()
        self.last_frame_time = time.time()
        self.last_aruco_time = 0.0
        self.corridor_width  = CORRIDOR_WIDTH

        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('🚀 FusionDriver 啟動！跑道優化版上線。')

    # =========================================================================
    # scan_callback：更細緻的角度分區
    # =========================================================================
    def scan_callback(self, msg):
        def get_min(indices):
            valid = [msg.ranges[i] for i in indices
                     # 修正：上限從 2.5 改為 1.5m，過濾遠距雜訊，專注近距障礙物
                     # 下限從 0.05 改為 0.03，避免極近距離被過濾掉
                     if i < len(msg.ranges) and 0.03 < msg.ranges[i] < 1.5]
            return min(valid) if valid else 10.0

        n = len(msg.ranges)

        # 正前方 ±25°（擴大，避免低矮海綿障礙物落在盲區）
        front_idx = list(range(0, 26)) + list(range(n - 26, n))
        # 左前 25~60°
        front_left_idx  = list(range(26, 61))
        # 右前 -60~-25°
        front_right_idx = list(range(n - 61, n - 26))
        # 左側 60~100°
        left_idx  = list(range(61, 101))
        # 右側 -100~-61°
        right_idx = list(range(n - 101, n - 61))

        self.front_dist       = get_min(front_idx)
        self.dist_front_left  = get_min(front_left_idx)
        self.dist_front_right = get_min(front_right_idx)
        self.dist_left        = get_min(left_idx)
        self.dist_right       = get_min(right_idx)

        # 只設旗標，不在此 publish
        if self.front_dist < EMERGENCY_STOP_DIST:
            if not self._emergency_stop_flag:
                self._emergency_stop_flag = True
                self.system_state = "EMERGENCY STOP"
                self.get_logger().warn(f"!!! 緊急停止：前方 {self.front_dist:.2f}m !!!")
        else:
            if self._emergency_stop_flag:
                self._emergency_stop_flag = False
                self.get_logger().info("障礙物已解除，恢復行駛")

    # =========================================================================
    # image_callback
    # =========================================================================
    def image_callback(self, msg):
        global output_frame

        # 終點永久停止：只更新畫面，不做任何其他處理
        if self.race_finished:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv_image = cv2.resize(cv_image, (FRAME_WIDTH, FRAME_HEIGHT))
                cv2.putText(cv_image, "🏁 RACE FINISHED - STOPPED", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                with lock:
                    output_frame = cv_image
            except Exception:
                pass
            return

        current_time = time.time()
        self.last_frame_time = current_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, (FRAME_WIDTH, FRAME_HEIGHT))
        except Exception:
            self.get_logger().error("影像轉換失敗！")
            return

        height, width, _ = cv_image.shape
        debug_img = cv_image.copy()

        # ======================================================================
        # 模組 1: ArUco 號誌辨識
        # ======================================================================
        if current_time - self.last_aruco_time > ARUCO_DETECT_INTERVAL:
            self.last_aruco_time = current_time
            corners, ids, _ = aruco.detectMarkers(
                cv_image, self.aruco_dict, parameters=self.aruco_params
            )
            if ids is not None:
                aruco.drawDetectedMarkers(debug_img, corners, ids)
                action_map = {
                    0: "STOP",
                    1: "TURN_LEFT_90",   # 對調：原 RIGHT → LEFT
                    2: "TURN_RIGHT_90",  # 對調：原 LEFT  → RIGHT
                    3: "STRAIGHT",
                    4: "COMBO_SHARP_S",
                    5: "AVOIDING",
                    6: "TURN_RIGHT_120",
                    7: "RANDOM_FORK",
                }
                for marker_id in ids.flatten():
                    new_action = action_map.get(int(marker_id), "NONE")
                    if (new_action != "NONE"
                            and new_action != self.planned_action
                            and new_action != self.current_action):
                        self.planned_action = new_action
                        self.plan_time = current_time
                        self.get_logger().info(f"偵測號誌: {new_action} (ID:{marker_id})")

        # ======================================================================
        # 模組 1.5: 延遲觸發
        # ======================================================================
        if self.planned_action != "NONE":
            if (current_time - self.plan_time) >= self.delay_time:
                self.current_action    = self.planned_action
                self.action_start_time = current_time
                self.planned_action    = "NONE"

                # 三連彎初始化
                self.s_bend_current_stage = 0
                self.s_bend_stage_start   = current_time

                # 持續時間設定
                if self.current_action == "STOP":
                    self.current_action_duration = STOP_DURATION      # -1 = 永久
                elif self.current_action == "COMBO_SHARP_S":
                    self.current_action_duration = sum(S_BEND_DURATIONS)
                elif self.current_action == "AVOIDING":
                    self.current_action_duration = AVOIDING_DURATION
                elif self.current_action in ("TURN_RIGHT_90", "TURN_LEFT_90"):
                    self.current_action_duration = TURN_90_DURATION
                elif self.current_action == "TURN_RIGHT_120":
                    self.current_action_duration = TURN_120_DURATION
                elif self.current_action == "RANDOM_FORK":
                    self.current_action = random.choice(["TURN_LEFT_90", "TURN_RIGHT_90"])
                    self.current_action_duration = TURN_90_DURATION
                else:
                    self.current_action_duration = 0.0

                self.get_logger().info(
                    f"觸發: {self.current_action} "
                    f"({'永久' if self.current_action_duration < 0 else f'{self.current_action_duration:.1f}s'})"
                )

        # ======================================================================
        # 模組 2: 車道線偵測 + 動作遮罩
        # ======================================================================
        scan_top    = int(height * 0.50)
        scan_bottom = int(height * 0.85)
        crop_img    = cv_image[scan_top:scan_bottom, 0:width]

        cv2.rectangle(debug_img, (0, scan_top), (width, scan_bottom), (255, 255, 0), 2)

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        # 黃線（跑道邊界）+ 白線（中線）都偵測
        mask_yellow = cv2.inRange(hsv, np.array([18,  80,  80]),  np.array([35,  255, 255]))
        mask_white  = cv2.inRange(hsv, np.array([0,   0,   160]), np.array([180, 50,  255]))
        mask = cv2.bitwise_or(mask_yellow, mask_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # === 動作遮罩覆蓋 ===
        action_label = ""

        if self.current_action != "NONE":
            elapsed = current_time - self.action_start_time
            # 永久停止判斷
            is_forever = (self.current_action_duration < 0)
            time_up = (not is_forever) and (elapsed >= self.current_action_duration)

            if self.current_action == "COMBO_SHARP_S":
                # 三連彎：每個 stage 有自己的計時
                stage_elapsed = current_time - self.s_bend_stage_start

                if self.s_bend_current_stage == 0:
                    if stage_elapsed < S_BEND_DURATIONS[1]:
                        mask[:, 0:int(width * 0.5)] = 0   # 遮左→右轉
                        action_label = "S-Bend [1/3] RIGHT"
                    else:
                        self.s_bend_current_stage = 1
                        self.s_bend_stage_start   = current_time
                        self.get_logger().info("S-Bend: Stage1 完成 → Stage2")

                elif self.s_bend_current_stage == 1:
                    if stage_elapsed < S_BEND_DURATIONS[2]:
                        mask[:, int(width * 0.5):width] = 0   # 遮右→左轉
                        action_label = "S-Bend [2/3] LEFT"
                    else:
                        self.s_bend_current_stage = 2
                        self.s_bend_stage_start   = current_time
                        self.get_logger().info("S-Bend: Stage2 完成 → Stage3")

                elif self.s_bend_current_stage == 2:
                    if stage_elapsed < S_BEND_DURATIONS[3]:
                        mask[:, 0:int(width * 0.5)] = 0   # 遮左→右轉
                        action_label = "S-Bend [3/3] RIGHT"
                    else:
                        finished = self.current_action
                        self.current_action = "NONE"
                        self.s_bend_current_stage = 0
                        self.get_logger().info(f"動作 {finished} 完成")

            elif time_up:
                # 一般動作時間到 → 啟動 POST_TURN 補償（僅限轉彎動作）
                finished = self.current_action
                if finished in ("TURN_LEFT_90", "TURN_RIGHT_90", "TURN_RIGHT_120"):
                    self.post_turn_active    = True
                    self.post_turn_start     = current_time
                    # 補償方向與主彎相同
                    if finished == "TURN_LEFT_90":
                        self.post_turn_direction = 1.0   # 左
                    else:
                        self.post_turn_direction = -1.0  # 右
                    self.get_logger().info(
                        f"動作 {finished} 完成 → 啟動 POST_TURN 補償"
                        f"({'左' if self.post_turn_direction > 0 else '右'})"
                    )
                else:
                    self.get_logger().info(f"動作 {finished} 完成")
                self.current_action = "NONE"
                self.s_bend_current_stage = 0

            else:
                # 一般動作執行中
                if self.current_action == "TURN_LEFT_90":
                    mask[:, int(width * 0.5):width] = 0
                    action_label = "FORCE LEFT 90°"

                elif self.current_action == "TURN_RIGHT_90":
                    mask[:, 0:int(width * 0.5)] = 0
                    action_label = "FORCE RIGHT 90°"

                elif self.current_action == "TURN_RIGHT_120":
                    # 更激進：遮蔽左側 (1-TURN_120_MASK_RATIO) 比例
                    mask[:, 0:int(width * TURN_120_MASK_RATIO)] = 0
                    action_label = "FORCE RIGHT 120°"

                elif self.current_action == "STRAIGHT":
                    # 保持居中：稍微遮兩側
                    mask[:, 0:int(width * 0.15)] = 0
                    mask[:, int(width * 0.85):width] = 0
                    action_label = "STRAIGHT"

                elif self.current_action == "AVOIDING":
                    # 避障：不遮罩，讓雷達排斥力主導
                    action_label = "AVOIDING"

                elif self.current_action == "STOP":
                    # 永久停止
                    if not self.race_finished:
                        self.race_finished = True
                        self.get_logger().info("🏁 到達終點，永久停止！")
                    action_label = "RACE FINISHED"

        # POST_TURN 補償遮罩
        if self.post_turn_active and self.current_action == "NONE":
            pt_elapsed = current_time - self.post_turn_start
            if pt_elapsed < POST_TURN_DURATION:
                if self.post_turn_direction > 0:   # 左補償：遮蔽右側
                    mask[:, int(width * 0.55):width] = 0
                    action_label = "POST_TURN (Left trim)"
                else:                              # 右補償：遮蔽左側
                    mask[:, 0:int(width * 0.45)] = 0
                    action_label = "POST_TURN (Right trim)"
            else:
                self.post_turn_active = False
                self.get_logger().info("POST_TURN 補償完成")

        # pending 提示
        if self.planned_action != "NONE":
            tl = self.delay_time - (current_time - self.plan_time)
            cv2.putText(debug_img, f"PENDING: {self.planned_action} in {tl:.1f}s",
                        (10, scan_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # ======================================================================
        # 車道中心計算
        # ======================================================================
        projection = np.any(mask > 0, axis=0)
        center_x   = width // 2
        left_bound  = None
        right_bound = None

        left_indices = np.where(projection[0:center_x])[0]
        if len(left_indices) > 0:
            left_bound = int(left_indices[-1])

        right_indices = np.where(projection[center_x:width])[0]
        if len(right_indices) > 0:
            right_bound = int(right_indices[0]) + center_x

        target_cx = None

        if left_bound is not None and right_bound is not None:
            target_cx  = (left_bound + right_bound) // 2
            lane_width = right_bound - left_bound
            # 寬車道稍微修正，避免壓內線
            if lane_width > self.corridor_width * 0.8:
                target_cx += int(lane_width * 0.04)

        elif left_bound is not None:
            if self.current_action in ("TURN_LEFT_90", "TURN_RIGHT_90",
                                       "TURN_RIGHT_120", "COMBO_SHARP_S"):
                target_cx = left_bound + int(width * 0.35)
            else:
                target_cx = left_bound + (self.corridor_width // 2)

        elif right_bound is not None:
            if self.current_action in ("TURN_LEFT_90", "TURN_RIGHT_90",
                                       "TURN_RIGHT_120", "COMBO_SHARP_S"):
                target_cx = right_bound - int(width * 0.35)
            else:
                target_cx = right_bound - (self.corridor_width // 2)

        draw_y = int((scan_top + scan_bottom) / 2)

        if left_bound is not None:
            cv2.line(debug_img, (left_bound, scan_top), (left_bound, scan_bottom), (0, 80, 255), 3)
        if right_bound is not None:
            cv2.line(debug_img, (right_bound, scan_top), (right_bound, scan_bottom), (0, 80, 255), 3)

        if target_cx is not None:
            self.line_error    = target_cx - (width / 2)
            self.line_detected = True
            self.last_seen_time = time.time()
            cv2.circle(debug_img, (target_cx, draw_y), 10, (0, 255, 0), -1)
            cv2.line(debug_img, (width // 2, draw_y), (target_cx, draw_y), (0, 255, 100), 2)
        else:
            self.line_detected = False

        # ======================================================================
        # HUD
        # ======================================================================
        overlay = debug_img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, debug_img, 0.45, 0, debug_img)

        state_color = {
            "CRUISING":        (0, 255, 0),
            "OBSTACLE STOP":   (0, 0, 255),
            "EMERGENCY STOP":  (0, 0, 255),
            "MEMORY TURN":     (0, 165, 255),
            "SEARCHING":       (0, 0, 255),
            "SIGN STOP":       (0, 200, 255),
            "POST_TURN":       (255, 200, 0),
            "AVOIDING":        (0, 140, 255),
            "AVOIDING-ENTRY":  (0, 220, 200),   # 進場緩衝：青色
            "AVOIDING-DANGER": (0, 0, 255),     # 危險閃避：紅色
        }.get(self.system_state, (200, 200, 200))

        cv2.putText(debug_img, f"SYS:{self.system_state}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
        cv2.putText(debug_img, f"SPD:{self.current_speed:.2f} STR:{self.current_steer:.2f}",
                    (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 避障時 Lidar 顯示改紅色；危險距離內的數值特別標記
        lidar_color = (0, 80, 255) if self.current_action == "AVOIDING" else (0, 255, 255)
        lidar_text = (f"L:{self.dist_left:.2f} FL:{self.dist_front_left:.2f} "
                      f"F:{self.front_dist:.2f} FR:{self.dist_front_right:.2f} R:{self.dist_right:.2f}")
        cv2.putText(debug_img, lidar_text, (5, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, lidar_color, 1)
        cv2.putText(debug_img, f"ACT:{self.current_action}  {action_label}",
                    (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        # 轉向指示條
        bx = int(width * 0.85)
        by = 30
        cv2.line(debug_img, (bx - 45, by), (bx + 45, by), (80, 80, 80), 3)
        cv2.circle(debug_img, (bx, by), 3, (200, 200, 200), -1)
        spx = max(-45, min(45, int(self.current_steer * -28)))
        cv2.circle(debug_img, (bx + spx, by), 7, (0, 255, 255), -1)

        with lock:
            output_frame = debug_img

    # =========================================================================
    # 模組 4: 控制迴路
    # =========================================================================
    def control_loop(self):
        cmd = Twist()

        # 0. 終點永久停止
        if self.race_finished:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.publisher_.publish(cmd)
            return

        # 1. 緊急停止（Lidar 觸發）
        if self._emergency_stop_flag:
            self.system_state = "EMERGENCY STOP"
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.publisher_.publish(cmd)
            return

        # 2. 障礙物停止
        if self.front_dist < OBSTACLE_STOP_DIST and self.current_action != "AVOIDING":
            self.system_state  = "OBSTACLE STOP"
            self.current_speed = 0.0
            self.current_steer = 0.0
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.publisher_.publish(cmd)
            return

        # 3. 號誌停止（STOP 動作）
        if self.current_action == "STOP":
            self.system_state  = "SIGN STOP"
            self.current_speed = 0.0
            self.current_steer = 0.0
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.publisher_.publish(cmd)
            return

        # 4. 正常行駛
        if self.line_detected:
            self.system_state = "CRUISING"
            error      = float(self.line_error)
            derivative = error - self.prev_error

            # PID（加入積分項）
            kp = 0.0035
            ki = 0.0001
            kd = 0.006

            # 積分項限幅，避免 windup
            self.integral_error = max(-200.0, min(200.0, self.integral_error + error))

            base_steer = -(error * kp + self.integral_error * ki + derivative * kd)
            self.prev_error = error

            target_speed = max(0.07, 0.18 - abs(base_steer) * 0.08)
            steer = base_steer

            # === 各動作行為 ===

            if self.current_action == "AVOIDING":
                # ==============================================================
                # 避障策略（針對此跑道：障礙物偏左，預設往右繞）
                #
                # 四段邏輯：
                #  0. 進場緩衝（前 ENTRY_TIME 秒）：強制減速，直線慢行穩定車身
                #  1. 危險距離（< DANGER）         ：強制大角度閃避，速度最低
                #  2. 安全距離（< SAFE）           ：雷達排斥力融合循跡，動態修正
                #  3. 無障礙                       ：保持預設右偏，緩慢通過
                # ==============================================================
                self.integral_error = 0.0  # 避障時重置積分

                elapsed_avoiding = time.time() - self.action_start_time

                # ── 階段 0：進場緩衝減速 ─────────────────────────────────────
                if elapsed_avoiding < AVOIDING_ENTRY_TIME:
                    # 線性從目前速度降到 AVOIDING_ENTRY_SPEED
                    ratio = elapsed_avoiding / AVOIDING_ENTRY_TIME
                    target_speed = max(AVOIDING_ENTRY_SPEED,
                                       0.18 - (0.18 - AVOIDING_ENTRY_SPEED) * ratio)
                    steer = base_steer * 0.5   # 減速段保持直行，輕微跟線
                    self.system_state = "AVOIDING-ENTRY"
                    self.get_logger().info(
                        f"避障進場緩衝: {elapsed_avoiding:.1f}/{AVOIDING_ENTRY_TIME:.1f}s "
                        f"速度={target_speed:.3f}"
                    )

                else:
                    # ── 各方位障礙距離 ────────────────────────────────────────
                    f  = self.front_dist
                    fl = self.dist_front_left
                    fr = self.dist_front_right
                    l  = self.dist_left
                    r  = self.dist_right

                    # ── 階段 1：危險距離強制閃避 ──────────────────────────────
                    danger = (f  < AVOIDING_DANGER_DIST or
                              fl < AVOIDING_DANGER_DIST or
                              fr < AVOIDING_DANGER_DIST)

                    if danger:
                        target_speed = AVOIDING_SPEED * 0.5   # 速度砍到最低
                        left_threat  = min(fl, l, f if fl < fr else 10.0)
                        right_threat = min(fr, r, f if fr < fl else 10.0)
                        if left_threat < right_threat:
                            steer = -1.3   # 左邊更危險 → 強制右閃
                        else:
                            steer = 1.3    # 右邊更危險 → 強制左閃
                        self.system_state = "AVOIDING-DANGER"
                        self.get_logger().warn(
                            f"⚠️ 危險！F:{f:.2f} FL:{fl:.2f} FR:{fr:.2f} "
                            f"→ 強制{'右' if steer < 0 else '左'}閃"
                        )

                    else:
                        # ── 階段 2/3：排斥力融合 ──────────────────────────────
                        repulsion = AVOIDING_DEFAULT_BIAS  # 預設右偏

                        if l < AVOIDING_SAFE_DIST:
                            repulsion -= (AVOIDING_SAFE_DIST - l) * AVOIDING_REPULSION
                        if fl < AVOIDING_SAFE_DIST:
                            repulsion -= (AVOIDING_SAFE_DIST - fl) * AVOIDING_REPULSION * 1.4
                        if r < AVOIDING_SAFE_DIST:
                            repulsion += (AVOIDING_SAFE_DIST - r) * AVOIDING_REPULSION
                        if fr < AVOIDING_SAFE_DIST:
                            repulsion += (AVOIDING_SAFE_DIST - fr) * AVOIDING_REPULSION * 1.4

                        max_rep    = AVOIDING_SAFE_DIST * AVOIDING_REPULSION * 1.4
                        normalized = max(-1.0, min(1.0, repulsion / max_rep)) * 0.9

                        steer = (base_steer * (1.0 - AVOIDING_TURN_BIAS)
                                 + normalized * AVOIDING_TURN_BIAS)
                        steer        = max(-1.5, min(1.5, steer))
                        target_speed = AVOIDING_SPEED   # 全程鎖定慢速
                        self.system_state = "AVOIDING"
                        self.get_logger().info(
                            f"避障: L:{l:.2f} FL:{fl:.2f} F:{f:.2f} "
                            f"FR:{fr:.2f} R:{r:.2f} "
                            f"排斥={repulsion:.2f} 轉向={steer:.2f}"
                        )

            elif self.current_action == "TURN_RIGHT_90":
                target_speed = TURN_90_SPEED
                steer = TURN_90_STEER
                self.get_logger().info("執行 90° 右轉")

            elif self.current_action == "TURN_LEFT_90":
                target_speed = TURN_90_SPEED
                steer = -TURN_90_STEER  # 鏡像方向
                self.get_logger().info("執行 90° 左轉")

            elif self.current_action == "TURN_RIGHT_120":
                target_speed = TURN_120_SPEED
                steer = TURN_120_STEER
                self.get_logger().info("執行 120° 右轉")

            elif self.current_action == "STRAIGHT":
                target_speed = STRAIGHT_SPEED
                steer = base_steer * 0.75  # 直線稍微降低靈敏度避免晃動
                self.integral_error = 0.0  # 直線重置積分

            elif self.current_action == "COMBO_SHARP_S":
                target_speed = S_BEND_SPEED
                # 根據當前 stage 給基礎力道，與 image_callback 的 mask 配合
                if self.s_bend_current_stage in (0, 2):
                    steer = -1.0   # 右轉
                else:
                    steer = 1.0    # 左轉

            # POST_TURN 補償：主彎結束後的小轉修正
            if self.post_turn_active and self.current_action == "NONE":
                pt_elapsed = time.time() - self.post_turn_start
                if pt_elapsed < POST_TURN_DURATION:
                    target_speed = POST_TURN_SPEED
                    steer = POST_TURN_STEER * self.post_turn_direction
                    self.system_state = "POST_TURN"

            # 速度/轉向限幅
            cmd.linear.x  = max(0.0,  min(0.25, target_speed))
            cmd.angular.z = max(-2.0, min(2.0,  steer))

            self.current_steer     = cmd.angular.z
            self.current_speed     = cmd.linear.x
            self.deep_steer_memory = self.deep_steer_memory * 0.8 + self.current_steer * 0.2

        else:
            # 車道線丟失
            lost_duration = time.time() - self.last_seen_time

            if lost_duration < 1.5:
                # 短暫丟失：記憶轉向維持
                self.system_state = "MEMORY TURN"
                cmd.linear.x  = 0.07
                cmd.angular.z = self.deep_steer_memory
                self.get_logger().warn(f"車道線短暫丟失，記憶轉向: {self.deep_steer_memory:.2f}")
            elif lost_duration < 4.0:
                # 中等丟失：原地慢速搜尋
                self.system_state  = "SEARCHING"
                self.current_speed = 0.0
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.3 if self.deep_steer_memory >= 0 else -0.3
                self.get_logger().error("搜尋中（根據記憶方向旋轉）")
            else:
                # 長時間丟失：停止並警告
                self.system_state  = "SEARCHING"
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
                self.get_logger().error("車道線長時間丟失，停止等待人工介入")

        self.publisher_.publish(cmd)


# =============================================================================
# 啟動
# =============================================================================
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main(args=None):
    rclpy.init(args=args)
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()

    node = FusionDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("鍵盤中斷，發布停止指令...")
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        node.publisher_.publish(stop_cmd)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
