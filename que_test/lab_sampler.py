"""
LAB 颜色采样工具
运行后图像会保存到磁盘，供你在 PC 上用鼠标点击采样
或者直接在终端输出 ROI 区域的 LAB 统计值

用法1（泰山派上直接看统计）：
    python lab_sampler.py
    → 自动采样中心区域并打印 LAB 均值/标准差

用法2（采样后存图，拷到PC上用OpenCV鼠标点击）：
    python lab_sampler.py --save
    → 保存 frame.png，拷回PC后用 lab_sampler_pc.py 点击采样
"""
import cv2
import numpy as np
import sys

cap = cv2.VideoCapture(9, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] 等待摄像头就绪...")
for _ in range(30):
    ret, frame = cap.read()

if not ret:
    print("[ERROR] 无法读取摄像头！")
    cap.release()
    exit()

frame = frame[120:360, 170:470]
h, w = frame.shape[:2]
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

# ============================================================
# 自动统计：中心区域 + 全图网格
# ============================================================
print("\n" + "=" * 60)
print("  钢球 LAB 颜色采样报告")
print("=" * 60)
print(f"  ROI 尺寸: {w} x {h}")
print()

# 1. 全图 L/A/B 通道的范围
print("--- 全图 LAB 统计 ---")
for i, name in enumerate(["L", "A", "B"]):
    ch = lab[:, :, i]
    print(f"  {name}: min={ch.min():3d}  max={ch.max():3d}  mean={ch.mean():6.1f}  std={ch.std():5.1f}")

print()

# 2. 中心区域（假设钢球在画面中央附近）
# 取中心 40x40 区域
cy, cx = h // 2, w // 2
rgn = lab[cy-20:cy+20, cx-20:cx+20]
print("--- 中心 40×40 区域 LAB 统计（假设钢球在中央） ---")
for i, name in enumerate(["L", "A", "B"]):
    ch = rgn[:, :, i]
    print(f"  {name}: min={ch.min():3d}  max={ch.max():3d}  mean={ch.mean():6.1f}  std={ch.std():5.1f}")

# 3. 转成 OpenMV 格式的建议阈值
# OpenMV: L = OpenCV_L / 2.55, A = OpenCV_A - 128, B = OpenCV_B - 128
l_omv = rgn[:, :, 0].mean() / 2.55
a_omv = rgn[:, :, 1].mean() - 128
b_omv = rgn[:, :, 2].mean() - 128

l_std = rgn[:, :, 0].std() / 2.55
a_std = rgn[:, :, 1].std()
b_std = rgn[:, :, 2].std()

print()
print("--- 基于中心区域的建议 OpenMV 阈值 ---")
print(f"  亮球猜测值（均值±2倍标准差）：")
print(f"  L: {max(0, int(l_omv - 2*l_std))} ~ {min(100, int(l_omv + 2*l_std))}")
print(f"  A: {max(-128, int(a_omv - 2*a_std))} ~ {min(127, int(a_omv + 2*a_std))}")
print(f"  B: {max(-128, int(b_omv - 2*b_std))} ~ {min(127, int(b_omv + 2*b_std))}")
print()
print("  建议先试: b {:d} {:d} {:d} {:d} {:d} {:d}".format(
    max(0, int(l_omv - 2*l_std)),
    min(100, int(l_omv + 2*l_std)),
    max(-128, int(a_omv - 3*a_std)),
    min(127, int(a_omv + 3*a_std)),
    max(-128, int(b_omv - 3*b_std)),
    min(127, int(b_omv + 3*b_std)),
))
print("=" * 60)

# 4. 保存采样图
if "--save" in sys.argv or "-s" in sys.argv:
    # 在中心画个框标记采样区域
    annotated = frame.copy()
    cv2.rectangle(annotated, (cx-20, cy-20), (cx+20, cy+20), (0, 255, 255), 2)
    cv2.imwrite("frame_sample.png", annotated)
    print("\n[OK] 采样图已保存为 frame_sample.png")
    print("[INFO] 拷到PC后可运行 lab_sampler_pc.py 鼠标点击采样")

cap.release()
