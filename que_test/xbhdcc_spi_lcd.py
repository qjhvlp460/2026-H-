import cv2
import numpy as np
import time
import platform

# ==========================================
# 🖥️ 泰山派 ST7789 屏幕接线表 (基于引脚规划图)
# ==========================================
# 屏幕引脚     | 物理引脚编号 | 泰山派 GPIO/复用功能
# -------------|--------------|---------------------
# SDA (MOSI)   | 左侧 Pin 19  | GPIO4_C3 (SPI3_MOSI)
# SCL (CLK)    | 左侧 Pin 23  | GPIO4_C2 (SPI3_CLK)
# CS (片选)    | 右侧 Pin 24  | GPIO4_C6 (SPI3_CS0)
# DC (数据/命令)| 右侧 Pin 40  | GPIO3_B4
# RES (复位)   | 右侧 Pin 38  | GPIO3_B3
# VCC (电源)   | 左侧 Pin 17  | 3.3V 电源
# GND (地)     | 左侧 Pin 39  | GND 地
# ------------------------------------------
# *注：根据实际测试跑通的情况，代码默认参数采用了 DC=B4, RES=B3。
# 如果你后续把物理排线按表格重新插拔了，只需在实例化时修改参数即可。

# ==========================================
# 泰山派 40Pin 引脚编号速查字典
# ==========================================
GPIO_MAP = {
    # GPIO0 组
    "GPIO0_B5": 13, "GPIO0_B6": 14, "GPIO0_B7": 15,
    # GPIO1 组
    "GPIO1_A4": 36,
    # GPIO3 组 (A组)
    "GPIO3_A1": 97, "GPIO3_A2": 98, "GPIO3_A3": 99, "GPIO3_A4": 100,
    "GPIO3_A5": 101, "GPIO3_A6": 102, "GPIO3_A7": 103,
    # GPIO3 组 (B组)
    "GPIO3_B0": 104, "GPIO3_B1": 105, "GPIO3_B2": 106, "GPIO3_B3": 107,
    "GPIO3_B4": 108, "GPIO3_B5": 109, "GPIO3_B6": 110, "GPIO3_B7": 111,
    # GPIO3 组 (C组)
    "GPIO3_C0": 112, "GPIO3_C2": 114, "GPIO3_C3": 115, "GPIO3_C4": 116, "GPIO3_C5": 117,
    # GPIO4 组
    "GPIO4_C2": 146, "GPIO4_C3": 147, "GPIO4_C5": 149, "GPIO4_C6": 150,
}

class ST7789Streamer:
    """ST7789 SPI 屏幕推流类，接收 OpenCV 图像并实时显示"""
    
    # 默认参数适配了你当前实际跑通的物理接线
    def __init__(self, spi_dev="/dev/spidev3.0", dc_pin="GPIO3_B4", res_pin="GPIO3_B3", speed_hz=40000000):
        # 解析引脚编号
        dc_num = GPIO_MAP.get(dc_pin, dc_pin) if isinstance(dc_pin, str) else dc_pin
        res_num = GPIO_MAP.get(res_pin, res_pin) if isinstance(res_pin, str) else res_pin
        
        print(f"[ST7789] 正在初始化屏幕 (SPI: {spi_dev}, DC: {dc_num}, RES: {res_num})...")
        
        # 局部导入，防止在 Windows 上误导包导致崩溃
        try:
            from periphery import SPI, GPIO
        except ImportError:
            raise ImportError("未找到 periphery 库！请确保在 Linux 环境下运行，并已执行 pip install python-periphery")

        self.dc_pin = GPIO(dc_num, "out")
        self.res_pin = GPIO(res_num, "out")
        self.spi = SPI(spi_dev, 0, speed_hz)
        self.chunk_size = 4096  # 解决 Message too long 的核心参数
        
        self._init_display()
        print("[ST7789] 屏幕初始化完成，等待推流。")

    def _set_data_command(self, is_data):
        self.dc_pin.write(is_data)

    def _send_command(self, cmd):
        self._set_data_command(False)
        self.spi.transfer([cmd])

    def _send_data(self, data):
        self._set_data_command(True)
        self.spi.transfer(data)

    def _reset_screen(self):
        self.res_pin.write(False)
        time.sleep(0.1)
        self.res_pin.write(True)
        time.sleep(0.1)

    def _init_display(self):
        self._reset_screen()
        self._send_command(0x11) # Sleep Out
        time.sleep(0.12)
        self._send_command(0x21)
        time.sleep(0.12)
        
        self._send_command(0x3A) # Color Mode
        self._send_data([0x05])  # 16-bit RGB565
        
        self._send_command(0x29) # Display On
        time.sleep(0.1)

    def update_frame(self, img):
        """传入任意尺寸的 OpenCV BGR 图像，自动缩放并推送到屏幕"""
        if img is None:
            return
            
        # 1. 缩放为 240x240
        img_resized = cv2.resize(img, (240, 240))
        
        # 2. BGR 转 RGB565 并处理字节序
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        R = (img_rgb[..., 0] >> 3).astype(np.uint16) << 11
        G = (img_rgb[..., 1] >> 2).astype(np.uint16) << 5
        B = (img_rgb[..., 2] >> 3).astype(np.uint16)
        rgb565 = (R | G | B).byteswap().tobytes()
        
        # 3. 设置显示窗口 (240x240)
        self._send_command(0x2A) 
        self._send_data([0x00, 0x00, 0x00, 0xEF])
        self._send_command(0x2B)
        self._send_data([0x00, 0x00, 0x00, 0xEF]) 
        
        # 4. 发送写显存命令
        self._send_command(0x2C)
        self._set_data_command(True)
        
        # 5. 分块发送图像数据
        data = bytearray(rgb565)
        for i in range(0, len(data), self.chunk_size):
            self.spi.transfer(data[i:i+self.chunk_size])

    def stop(self):
        """释放硬件资源"""
        self.spi.close()
        self.dc_pin.close()
        self.res_pin.close()
        print("[ST7789] 硬件资源已释放。")

# --- 独立测试入口 ---
if __name__ == "__main__":
    # 如果直接运行此文件，将执行简单的颜色交替测试
    try:
        screen = ST7789Streamer()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # 红, 绿, 蓝 (BGR)
        print("开始屏幕测试，按 Ctrl+C 退出...")
        idx = 0
        while True:
            # 生成纯色测试图
            frame = np.zeros((240, 240, 3), dtype=np.uint8)
            frame[:] = colors[idx]
            cv2.putText(frame, "SPI LCD TEST", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            screen.update_frame(frame)
            idx = (idx + 1) % 3
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        if 'screen' in locals():
            screen.stop()