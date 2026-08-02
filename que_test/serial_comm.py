import time
import threading

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[Serial] pyserial not installed. Run: pip install pyserial")


class SerialSender:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, mock=False):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.lock = threading.Lock()
        self.total_sent = 0
        self.total_errors = 0
        self.mock = mock or (not HAS_SERIAL)

        if self.mock:
            print(f"[Serial] MOCK mode (no real serial)")
        else:
            self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"[Serial] Connected: {self.port} @ {self.baudrate} baud")
        except Exception as e:
            print(f"[Serial] Open failed: {e}, switching to MOCK")
            self.mock = True
            self.ser = None

    def send(self, x, y, r, status):
        if self.mock:
            self.total_sent += 1
            return
        if self.ser is None:
            return
        try:
            msg = f"{x},{y},{r},{status}\n"
            with self.lock:
                self.ser.write(msg.encode())
            self.total_sent += 1
        except Exception:
            self.total_errors += 1
            if self.total_errors < 3:
                print(f"[Serial] Send error #{self.total_errors}")
            if self.total_errors > 50:
                print("[Serial] Too many errors, switching to MOCK")
                self.mock = True
                self.ser = None

    def get_stats(self):
        return self.total_sent, self.total_errors

    def close(self):
        if self.ser:
            self.ser.close()
            print("[Serial] Closed")
