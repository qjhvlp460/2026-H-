import cv2
import time
import numpy as np
from xbhdcc_tools import detect_cameras, WebStreamer
from xbhdcc_spi_lcd import ST7789Streamer
from serial_comm import SerialSender

# ============================================================
# 原图尺寸 & 裁剪比例
# ============================================================
SRC_W, SRC_H = 640, 480
CROP_RATIO = 0.90  # 保留 90%

# 居中裁剪后的实际像素范围
CROP_W = int(SRC_W * CROP_RATIO)   # 576
CROP_H = int(SRC_H * CROP_RATIO)   # 432
CROP_X0 = (SRC_W - CROP_W) // 2    # 32  (原图坐标)
CROP_Y0 = (SRC_H - CROP_H) // 2    # 24  (原图坐标)
CROP_X1 = CROP_X0 + CROP_W         # 608
CROP_Y1 = CROP_Y0 + CROP_H         # 456

# 视野裁剪（原图坐标 → 传给 cap.read 后的裁剪用）
ROI_X0, ROI_Y0 = CROP_X0, CROP_Y0
ROI_X1, ROI_Y1 = CROP_X1, CROP_Y1

# ============================================================
# 所有框线用百分比定义（基于裁剪后坐标系 576 x 432）
# 修改百分比即可等比例缩放，不用手算像素
# ============================================================

# --- 检测门控矩形（黄框）---
# 占裁剪后画面的 90% 宽度 × 70% 高度，居中
GATE_W_RATIO = 0.90   # 宽度占比
GATE_H_RATIO = 0.10   # 高度占比
GATE_X0 = int(CROP_W * (1 - GATE_W_RATIO) / 2)            # 29
GATE_Y0 = int(CROP_H * (1 - GATE_H_RATIO) / 2)            # 65
GATE_X1 = int(CROP_W * (1 - (1 - GATE_W_RATIO) / 2))      # 547
GATE_Y1 = int(CROP_H * (1 - (1 - GATE_H_RATIO) / 2))      # 367

# --- 水管端线（青色）---
# 占裁剪后宽度的 84%，居中
PIPE_W_RATIO = 0.84
PIPE_LEFT_X  = int(CROP_W * (1 - PIPE_W_RATIO) / 2)        # 46
PIPE_RIGHT_X = int(CROP_W * (1 - (1 - PIPE_W_RATIO) / 2))  # 530
# 占裁剪后高度的 40%，垂直居中l
PIPE_H_RATIO = 0.08
PIPE_TOP_Y   = int(CROP_H * (1 - PIPE_H_RATIO) / 2)         # 130
PIPE_BOT_Y   = int(CROP_H * (1 - (1 - PIPE_H_RATIO) / 2))  # 302

# --- 派生参数 ---
PIPE_LENGTH_CM = 25.0                                    # 水管实际长度（题目给定）
PIPE_CENTER_X  = (PIPE_LEFT_X + PIPE_RIGHT_X) / 2        # 水管中心像素 x
PIPE_CENTER_Y  = (PIPE_TOP_Y + PIPE_BOT_Y) / 2           # 水管中心像素 y
CM_PER_PX      = PIPE_LENGTH_CM / (PIPE_RIGHT_X - PIPE_LEFT_X)

# ============================================================
# 颜色阈值转换（OpenMV 格式 → OpenCV LAB）
# ============================================================
def convert_lab_thresholds(openmv_thresholds):
    l_min, l_max, a_min, a_max, b_min, b_max = openmv_thresholds
    lower = np.array([int(l_min * 2.55), int(a_min + 128), int(b_min + 128)])
    upper = np.array([int(l_max * 2.55), int(a_max + 128), int(b_max + 128)])
    return lower, upper

# ============================================================
# α-β 跟踪滤波器（带速度预测：匀速运动零滞后，高速不跟丢）
# alpha 越大越跟手但越抖；beta 控制速度估计的响应快慢
# ============================================================
class AlphaBetaFilter:
    def __init__(self, alpha=0.6, beta=0.25):
        self.alpha = alpha
        self.beta = beta
        self.x = None   # 位置估计
        self.v = 0.0    # 速度估计（px/帧）

    def update(self, meas):
        meas = float(meas)
        if self.x is None:
            self.x = meas
            self.v = 0.0
        else:
            pred = self.x + self.v           # 先用速度预测
            residual = meas - pred           # 预测与实测的残差
            self.x = pred + self.alpha * residual
            self.v = self.v + self.beta * residual
        return int(round(self.x))

    def reset(self):
        self.x = None
        self.v = 0.0

# ============================================================
# 钢球检测器（核心类，与 ball_detect.py 一致）
# 解决两大问题：
#   1. 金属反光 → 球心空洞（闭运算填充 + 反光洞特征加分）
#   2. 环境噪点闪烁 → 误判（连通域滤波 + 多帧确认）
# ============================================================
class BallDetector:
    def __init__(self):
        # --- LAB 阈值（CanMV 阈值编辑器标定：银色钢球）---
        self.lower_lab, self.upper_lab = convert_lab_thresholds(
            [33, 57, -10, 10, -1, 20]
        )

        # --- 形态学核 ---
        self.kernel_tiny  = np.ones((3, 3), np.uint8)   # 去噪点/毛刺
        self.kernel_small = np.ones((5, 5), np.uint8)   # 填反光空洞

        # --- 筛选参数（钢球半径约 8~14px，裁剪后噪点少，适度放宽）---
        self.MIN_AREA   = 55       # 最小面积（最小球半圆 ≈ π·8²/2，再留余量）
        self.MAX_AREA   = 1000     # 最大面积（最大球整圆 ≈ π·14²，留更多余量）
        self.MIN_RADIUS = 4        # 最小半径（裁剪后允许更小/更远的球）
        self.MAX_RADIUS = 20       # 最大半径（放宽大球/近球上限）
        self.MIN_CIRC   = 0.28     # 降低圆形度门槛，允许更扁的半圆/反光

        # --- 抗噪参数 ---
        self.MAX_CANDIDATES = 3      # 最多保留几个候选轮廓
        self.FLICKER_FRAMES = 2      # 连续几帧出现才认可（裁剪后噪点少，可更快确认）
        self.FLICKER_SPREAD_MAX = 60 # 缓冲内位置跨度上限(px)，超过判为噪点闪烁（需容纳高速球）

        # --- 反光空洞加分（钢球中心反光特征）---
        self.HOLE_BONUS      = 0.2   # 确认反光空洞后的固定加分
        self.HOLE_RATIO_MIN  = 0.01  # 更小反光洞也算
        self.HOLE_RATIO_MAX  = 0.55  # 更大反光洞也接受
        self.HOLE_CENTER_MAX = 0.5   # 空洞质心须在拟合圆心 0.5r 以内

        # --- 状态 ---
        self.flicker_buffer = []   # 闪烁检测缓冲
        self.valid_center = None   # 上次有效中心
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = 12  # 裁剪后误报少，允许更长丢失保持

    # --------------------------------------------------------
    # 主入口：输入 BGR 帧，返回 (best_result, mask)
    # best_result = (cx, cy, r, is_half, circularity, score, has_hole)
    # --------------------------------------------------------
    def detect(self, frame):
        # ========== 步骤1：LAB 阈值提取 ==========
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, self.lower_lab, self.upper_lab)

        # ========== 步骤2：去噪 + 填反光空洞 ==========
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_tiny)    # 去小噪点
        mask_nofill = mask.copy()  # 留底：此刻反光黑洞尚未被填充（用于反光特征判断）
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_small)  # 填球心反光空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_tiny)    # 去闭运算毛刺

        # ========== 步骤3：连通域面积滤波（已知球尺寸，直接卡死范围） ==========
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        valid_components = []
        for i in range(1, num_labels):  # 跳过背景(0)
            area = stats[i, cv2.CC_STAT_AREA]
            if self.MIN_AREA <= area <= self.MAX_AREA:
                valid_components.append(i)

        # 候选过多时只保留面积最大的几个
        if len(valid_components) > self.MAX_CANDIDATES:
            areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in valid_components]
            areas.sort(key=lambda x: x[1], reverse=True)
            valid_components = [i for i, _ in areas[:self.MAX_CANDIDATES]]

        mask_clean = np.zeros_like(mask)
        for i in valid_components:
            mask_clean[labels == i] = 255

        # ========== 步骤4：查找轮廓并拟合圆 ==========
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_result = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # --- 圆形度 ---
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # --- 实心度（area / convex_hull_area）---
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # --- 宽高比（接近 1 表示圆形）---
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect = min(w_rect, h_rect) / max(w_rect, h_rect) if max(w_rect, h_rect) > 0 else 0

            # --- 最小外接圆 ---
            ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cx), int(cy), int(radius)

            if radius < self.MIN_RADIUS or radius > self.MAX_RADIUS:
                continue

            # --- 半圆检测 ---
            # 如果球被反光/遮挡，轮廓可能只有半圆
            # 策略：用凸包拟合 + 验证宽高比 + 降低圆形度要求
            is_half_circle = False
            if circularity < self.MIN_CIRC:
                # 检查是否像半圆：宽高比接近1 + 实心度在合理范围
                if aspect > 0.5 and 0.3 < solidity < 0.8:
                    is_half_circle = True
                    # 对半圆，用凸包的中心作为估计中心
                    M = cv2.moments(hull)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    # 半径从面积反推：半圆面积 = πr²/2
                    r = int(np.sqrt(2 * area / np.pi))

            # --- 反光空洞检测（钢球中心反光特征）---
            # 填充轮廓与闭运算前的留底 mask 对比：
            # 轮廓内、留底中为黑的像素 = 被白色完全包围的反光洞（边缘缺口不计）
            filled = np.zeros_like(mask_clean)
            cv2.drawContours(filled, [cnt], -1, 255, -1)
            hole_mask = cv2.bitwise_and(filled, cv2.bitwise_not(mask_nofill))
            hole_area = cv2.countNonZero(hole_mask)
            filled_area = cv2.countNonZero(filled)
            has_hole = False
            if hole_area > 0 and filled_area > 0:
                hole_ratio = hole_area / filled_area
                if self.HOLE_RATIO_MIN <= hole_ratio <= self.HOLE_RATIO_MAX:
                    # 空洞质心须在拟合圆心 0.5r 以内
                    Mh = cv2.moments(hole_mask, binaryImage=True)
                    hx = Mh["m10"] / Mh["m00"]
                    hy = Mh["m01"] / Mh["m00"]
                    if np.hypot(hx - cx, hy - cy) <= self.HOLE_CENTER_MAX * r:
                        has_hole = True

            # --- 综合评分 ---
            # 圆形度权重 0.4 + 实心度权重 0.3 + 宽高比权重 0.3
            score = (circularity * 0.4 +
                    solidity * 0.3 +
                    aspect * 0.3)

            # 半圆降权
            if is_half_circle:
                score *= 0.65  # 半圆可信度略低，但仍可竞争

            # 反光空洞固定加分（强特征：金属球中心反光）
            if has_hole:
                score += self.HOLE_BONUS

            if score > best_score:
                best_score = score
                best_result = (cx, cy, r, is_half_circle, circularity, score, has_hole)

        # ========== 步骤5：闪烁检测（多帧确认） ==========
        if best_result is not None:
            cx, cy, r, is_half, circ, score, has_hole = best_result
            self.flicker_buffer.append((cx, cy, r))
            if len(self.flicker_buffer) > self.FLICKER_FRAMES:
                self.flicker_buffer.pop(0)

            # 计算缓冲区中的一致性（所有候选与中位数的距离）
            if len(self.flicker_buffer) >= 2:
                xs = [p[0] for p in self.flicker_buffer]
                ys = [p[1] for p in self.flicker_buffer]
                spread = max(max(xs)-min(xs), max(ys)-min(ys))
                if spread > self.FLICKER_SPREAD_MAX:  # 位置跳跃太大，可能是噪点闪烁
                    # 清空脏数据，防止 buffer 污染导致持续误拒
                    self.flicker_buffer.clear()
                    best_result = None

        # ========== 步骤6：丢失处理 ==========
        if best_result is not None:
            was_lost = self.lost_frames > 0
            self.lost_frames = 0
            # 重新检测到球时清空旧缓冲区，防止旧位置干扰新位置
            if was_lost:
                self.flicker_buffer.clear()
            self.valid_center = (best_result[0], best_result[1], best_result[2])
        else:
            self.lost_frames += 1
            if self.lost_frames >= self.MAX_LOST_FRAMES:
                self.valid_center = None
                self.flicker_buffer.clear()

        # 返回最终 mask（用于调试显示）
        return best_result, mask_clean

# ============================================================
# 偏差计算：球心像素 x → 相对水管中心 O 的偏差（mm，右正左负）
# ============================================================
def calc_deviation_mm(ball_cx):
    return int(round((ball_cx - PIPE_CENTER_X) * CM_PER_PX * 10))

# ============================================================
# 水管标定可视化：黄色 ROI 框 + 青色端线 + 红色中心十字
# ============================================================
def draw_pipe_calibration(display):
    # 黄色检测门控框（识别只在此框内进行）
    cv2.rectangle(display, (GATE_X0, GATE_Y0), (GATE_X1, GATE_Y1), (0, 255, 255), 2)
    # 青色端线
    cv2.line(display, (PIPE_LEFT_X, PIPE_TOP_Y), (PIPE_LEFT_X, PIPE_BOT_Y), (255, 255, 0), 2)
    cv2.line(display, (PIPE_RIGHT_X, PIPE_TOP_Y), (PIPE_RIGHT_X, PIPE_BOT_Y), (255, 255, 0), 2)
    # 红色中心十字
    ccx = int(PIPE_CENTER_X)
    ccy = (PIPE_TOP_Y + PIPE_BOT_Y) // 2
    cv2.line(display, (ccx - 12, ccy), (ccx + 12, ccy), (0, 0, 255), 2)
    cv2.line(display, (ccx, ccy - 12), (ccx, ccy + 12), (0, 0, 255), 2)

# ============================================================
# 底部刻度条：-12.50 / 0 / +12.50，绿点 = 球位置，红标 = 中心
# ============================================================
def draw_scale_bar(display, dev_mm, ball_valid):
    h, w = display.shape[:2]
    bar_y = h - 22
    x0, x1 = int(w * 0.15), int(w * 0.85)
    cx0 = (x0 + x1) // 2

    def dev_to_x(mm):
        mm = max(-125, min(125, mm))   # 限制在 ±12.5cm 量程内
        return int(cx0 + mm / 125.0 * (x1 - x0) / 2)

    # 主尺
    cv2.line(display, (x0, bar_y), (x1, bar_y), (180, 180, 180), 1)
    # 三个刻度 + 文字
    for mm, label in [(-125, "-12.50"), (0, "0"), (125, "+12.50")]:
        x = dev_to_x(mm)
        cv2.line(display, (x, bar_y - 5), (x, bar_y + 5), (180, 180, 180), 1)
        cv2.putText(display, label, (x - 22, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    # 中心红标
    cv2.line(display, (cx0, bar_y - 8), (cx0, bar_y + 8), (0, 0, 255), 2)
    # 球位置绿点
    if ball_valid:
        cv2.circle(display, (dev_to_x(dev_mm), bar_y), 5, (0, 255, 0), -1)

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # detect_cameras()  # 调试时取消注释，查看可用摄像头索引

    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    streamer = WebStreamer(port=8081)
    lcd_streamer = ST7789Streamer()
    serial_sender = SerialSender(port='/dev/ttyUSB0', baudrate=115200)

    detector = BallDetector()

    # FPS 计算
    fps = 0.0
    fps_smooth = 0.0
    last_time = time.time()

    # α-β 跟踪滤波（x/y/r 各一个，高速低滞后）
    filter_x = AlphaBetaFilter()
    filter_y = AlphaBetaFilter()
    filter_r = AlphaBetaFilter()

    print("LAB Lower:", detector.lower_lab)
    print("LAB Upper:", detector.upper_lab)
    print(f"Pipe center px: {PIPE_CENTER_X}, cm/px: {CM_PER_PX:.4f}")
    print("[INFO] detect_new started. Press Ctrl+C to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ROI 裁剪
        frame = frame[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
        display = frame.copy()
        h, w = frame.shape[:2]

        # ==================== FPS ====================
        curr_time = time.time()
        dt = curr_time - last_time
        last_time = curr_time
        if dt > 0:
            fps = 1.0 / dt
        fps_smooth = fps_smooth * 0.9 + fps * 0.1

        # 左上角 FPS 显示（半透明底板）
        overlay = display.copy()
        cv2.rectangle(overlay, (5, 5), (260, 48), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cv2.putText(display, f"FPS: {fps_smooth:.1f}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # ==================== 水管标定可视化 ====================
        draw_pipe_calibration(display)

        # ==================== 钢球检测（仅在黄框内） ====================
        gate_frame = frame[GATE_Y0:GATE_Y1, GATE_X0:GATE_X1]
        result, mask_gate = detector.detect(gate_frame)

        # 掩码贴回显示坐标系（调试用）
        mask_debug = np.zeros((h, w), np.uint8)
        mask_debug[GATE_Y0:GATE_Y1, GATE_X0:GATE_X1] = mask_gate

        ball_valid = result is not None
        dev_mm = 0

        if ball_valid:
            cx, cy, r, is_half, circ, score, has_hole = result
            cx += GATE_X0   # 坐标映回显示坐标系
            cy += GATE_Y0

            # α-β 滤波（带速度预测，高速不滞后）
            fx = filter_x.update(cx)
            fy = filter_y.update(cy)
            fr = filter_r.update(r)

            # 偏差（mm，右正左负）→ 串口 x 字段
            dev_mm = calc_deviation_mm(fx)
            status_code = 2 if is_half else 1
            serial_sender.send(dev_mm, fy, fr, status_code)

            # 画圆
            color = (0, 255, 0) if not is_half else (0, 255, 255)  # 半圆用黄色
            cv2.circle(display, (fx, fy), fr, color, 2)
            # 圆心十字
            cv2.line(display, (fx - 15, fy), (fx + 15, fy), color, 2)
            cv2.line(display, (fx, fy - 15), (fx, fy + 15), color, 2)
            # 圆心坐标
            cv2.putText(display, f"({fx},{fy})", (fx + 18, fy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            # 半径 + 评分
            mode_str = "HALF" if is_half else "FULL"
            if has_hole:
                mode_str += "+H"   # 检测到中心反光空洞
            info = f"R={fr} C={circ:.2f} S={score:.2f} [{mode_str}]"
            cv2.putText(display, info, (fx + 18, fy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            # 偏差值大字显示（与 LOST 横幅互斥，不会重叠）
            cv2.putText(display, f"DEV:{dev_mm:+d}mm", (15, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        else:
            filter_x.reset()
            filter_y.reset()
            filter_r.reset()
            serial_sender.send(0, 0, 0, 0)
            # 红色 "LOST" 提示
            overlay2 = display.copy()
            cv2.rectangle(overlay2, (5, 50), (220, 90), (0, 0, 200), -1)
            cv2.addWeighted(overlay2, 0.6, display, 0.4, 0, display)
            cv2.putText(display, "BALL LOST", (15, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ==================== 底部刻度条 ====================
        draw_scale_bar(display, dev_mm, ball_valid)

        # ==================== 调试信息 ====================
        # 右上角显示检测状态
        status_text = f"Lost:{detector.lost_frames}/{detector.MAX_LOST_FRAMES}"
        cv2.putText(display, status_text, (w - 280, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # ==================== 投流 ====================
        # 通道0：标注结果
        streamer.update_frame(0, display)
        # 通道1：二值掩码（黑白显示）
        streamer.update_frame(1, mask_debug)
        # LCD
        lcd_streamer.update_frame(display)
