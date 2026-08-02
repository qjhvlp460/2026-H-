"""
LAB 阈值标定工具
用法：在泰山派上运行此脚本，浏览器打开 http://<IP>:8080 即可实时看效果
在终端输入命令调整阈值，调好后阈值自动保存到 ball_detect.py
"""
import cv2
import numpy as np
import re
import os

# ============================================================
# 默认阈值（OpenMV 格式：L 0-100, A -128~127, B -128~127）
# ============================================================
class Params:
    def __init__(self):
        # 亮球阈值
        self.bright = [84, 100, -104, 127, -52, 118]
        # 阴影阈值
        self.shadow = [15, 55, -5, 5, -5, 5]

P = Params()

# ============================================================
# 将 OpenMV 阈值转为 OpenCV LAB 阈值
# ============================================================
def omv_to_ocv(omv):
    l_min, l_max, a_min, a_max, b_min, b_max = omv
    lower = np.array([int(l_min * 2.55), int(a_min + 128), int(b_min + 128)])
    upper = np.array([int(l_max * 2.55), int(a_max + 128), int(b_max + 128)])
    return lower, upper

# ============================================================
# 应用阈值并生成调试画面
# ============================================================
def process(frame, bright_omv, shadow_omv):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 亮球
    lo_b, hi_b = omv_to_ocv(bright_omv)
    mask_bright = cv2.inRange(lab, lo_b, hi_b)

    # 阴影
    lo_s, hi_s = omv_to_ocv(shadow_omv)
    mask_shadow = cv2.inRange(lab, lo_s, hi_s)

    # 合并
    mask = cv2.bitwise_or(mask_bright, mask_shadow)

    # 构建三通道显示（原始图 + 亮球mask + 阴影mask + 合并mask）
    h, w = frame.shape[:2]
    mask_bgr_bright = cv2.cvtColor(mask_bright, cv2.COLOR_GRAY2BGR)
    mask_bgr_shadow = cv2.cvtColor(mask_shadow, cv2.COLOR_GRAY2BGR)
    mask_bgr_comb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 标注文字
    for name, img in [("Bright", mask_bgr_bright), ("Shadow", mask_bgr_shadow), ("Combined", mask_bgr_comb)]:
        cv2.putText(img, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 拼接：原始 | 亮球 | 阴影 | 合并
    top = np.hstack([frame, mask_bgr_bright])
    bot = np.hstack([mask_bgr_shadow, mask_bgr_comb])
    panel = np.vstack([top, bot])

    return mask, panel


# ============================================================
# 打印帮助信息和当前阈值
# ============================================================
def print_status():
    print("\n" + "=" * 60)
    print("  LAB 阈值标定工具 - 命令列表")
    print("=" * 60)
    print("  b Lmin Lmax Amin Amax Bmin Bmax  → 设置亮球(Bright)阈值")
    print("  s Lmin Lmax Amin Amax Bmin Bmax  → 设置阴影(Shadow)阈值")
    print("  p                                 → 打印当前阈值")
    print("  save                              → 保存到 ball_detect.py")
    print("  q                                 → 退出")
    print("=" * 60)
    print(f"\n当前亮球阈值: {P.bright}")
    print(f"当前阴影阈值: {P.shadow}")
    print(f"\n阈值范围: L(0~100), A(-128~127), B(-128~127)")
    print("示例: b 84 100 -104 127 -52 118")
    print()

# ============================================================
# 保存阈值到 ball_detect.py
# ============================================================
def save_thresholds():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(script_dir, "ball_detect.py")
    if not os.path.exists(target):
        print(f"[ERROR] 找不到 {target}")
        return

    with open(target, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换亮球阈值
    old_bright = re.search(r'\[(\d+),\s*(\d+),\s*(-?\d+),\s*(\d+),\s*(-?\d+),\s*(\d+)\]', content)
    if old_bright:
        new_bright = f"[{P.bright[0]}, {P.bright[1]}, {P.bright[2]}, {P.bright[3]}, {P.bright[4]}, {P.bright[5]}]"
        content = content.replace(old_bright.group(0), new_bright, 1)

    # 替换阴影阈值（第二次匹配）
    old_shadow = re.search(r'\[(\d+),\s*(\d+),\s*(-?\d+),\s*(\d+),\s*(-?\d+),\s*(\d+)\]', content)
    if old_shadow:
        new_shadow = f"[{P.shadow[0]}, {P.shadow[1]}, {P.shadow[2]}, {P.shadow[3]}, {P.shadow[4]}, {P.shadow[5]}]"
        content = content.replace(old_shadow.group(0), new_shadow, 1)

    with open(target, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] 阈值已保存到 {target}")
    print(f"     亮球: {P.bright}")
    print(f"     阴影: {P.shadow}")

# ============================================================
# 解析用户输入的阈值
# ============================================================
def parse_threshold(args):
    if len(args) != 6:
        print("[ERROR] 需要6个参数: Lmin Lmax Amin Amax Bmin Bmax")
        return None
    try:
        vals = [int(x) for x in args]
    except ValueError:
        print("[ERROR] 参数必须是整数")
        return None
    return vals

# ============================================================
# 主循环
# ============================================================
if __name__ == "__main__":
    # 导入同目录的推流模块
    from xbhdcc_tools import WebStreamer
    from xbhdcc_spi_lcd import ST7789Streamer

    import threading

    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    streamer = WebStreamer(port=8081)
    lcd = ST7789Streamer()

    print_status()

    stop = False

    def input_thread():
        global stop
        while not stop:
            try:
                cmd = input().strip().split()
                if not cmd:
                    continue
                c = cmd[0].lower()
                if c == 'q':
                    stop = True
                    print("[INFO] 正在退出...")
                elif c == 'b':
                    vals = parse_threshold(cmd[1:])
                    if vals:
                        P.bright = vals
                        print(f"[OK] 亮球阈值更新为: {vals}")
                elif c == 's':
                    vals = parse_threshold(cmd[1:])
                    if vals:
                        P.shadow = vals
                        print(f"[OK] 阴影阈值更新为: {vals}")
                elif c == 'p':
                    print(f"\n亮球: {P.bright}")
                    print(f"阴影: {P.shadow}")
                elif c == 'save':
                    save_thresholds()
                else:
                    print(f"[ERROR] 未知命令: {c}")
            except EOFError:
                break

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    print("\n[INFO] 浏览器打开 http://<IP>:8081 查看实时标定画面")
    print("[INFO] 通道0 = 原始+二值化面板 | 通道1 = 合并mask（伪彩色）\n")

    while not stop:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = frame[120:360, 170:470]   # ROI

        mask, panel = process(frame, P.bright, P.shadow)

        # 通道0：四格面板
        streamer.update_frame(0, panel)
        # 通道1：合并mask伪彩色
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        streamer.update_frame(1, mask_color)
        # LCD
        lcd.update_frame(panel)

    cap.release()
    streamer.stop()
    lcd.stop()
    print("[INFO] 标定工具已退出。")
