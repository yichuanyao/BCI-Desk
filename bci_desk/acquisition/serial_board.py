"""921600 等非 BrainFlow 默认波特率下的 Cyton+Daisy 串口采集（OpenBCI 帧格式）。

Cyton+Daisy 交错（与 BrainFlow cyton_daisy.cpp 一致）：
    偶数 sample_id → Daisy CH9–16（半帧先到达，暂存）
    奇数 sample_id → Cyton CH1–8（半帧后到达，拼满 16 通道后提交）
每 2 个 UART 包拼 1 个 16 通道样本 → 通道 Hz = 板卡帧 Hz / 2。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import serial
from brainflow.board_shim import BoardShim


class SerialBoardError(RuntimeError):
    """pyserial 板卡路径专用异常（勿用 BrainFlowError，其 exit_code 须为合法枚举值）。"""


START_BYTE = 0xA0
END_BYTE_MIN = 0xC0
END_BYTE_MAX = 0xC6
DEFAULT_GAIN = 24
EEG_SCALE = 4.5 / float((2**23) - 1) / DEFAULT_GAIN * 1_000_000.0


def _cast_24bit_to_int32(data: bytes) -> int:
    value = (data[0] << 16) | (data[1] << 8) | data[2]
    if value & 0x800000:
        value -= 0x1000000
    return value


class PySerialCytonDaisyBoard:
    """与 BrainFlow BoardShim 最小兼容的 Cyton+Daisy 串口板卡。"""

    def __init__(self, port: str, baud_rate: int, board_id: int):
        self.port = str(port)
        self.baud_rate = int(baud_rate)
        self.board_id = int(board_id)
        self.num_rows = BoardShim.get_num_rows(board_id)
        self.eeg_channels = list(BoardShim.get_eeg_channels(board_id))
        self.timestamp_channel = BoardShim.get_timestamp_channel(board_id)

        self._ser: Optional[serial.Serial] = None
        self._initialized = False
        self._streaming = False
        self._keep_alive = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending: Deque[np.ndarray] = deque()
        self._history: Deque[np.ndarray] = deque(maxlen=50000)
        self._partial: Optional[np.ndarray] = None
        self._samples_total = 0
        self._uart_packets_total = 0

    @property
    def stream_samples_total(self) -> int:
        """已解析的完整 16 通道样本总数（不受 get_board_data 清空 pending 影响）。"""
        with self._lock:
            return int(self._samples_total)

    @property
    def stream_uart_packets_total(self) -> int:
        """已解析的 UART 33 字节包总数（Cyton+Daisy 约为样本率 × 2）。"""
        with self._lock:
            return int(self._uart_packets_total)

    def prepare_session(self):
        if self._initialized:
            return
        if not self.port:
            raise SerialBoardError("串口名为空。")
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=2.0,
            )
        except serial.SerialException as exc:
            raise SerialBoardError(
                f"无法打开 {self.port} @ {self.baud_rate}：{exc}"
            ) from exc

        time.sleep(0.3)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        # 若已在采集，先停流再软复位，避免 921600 下缓冲里是二进制包导致读不到 $$$
        self._try_write(b"s", pause_s=0.15)
        self._ser.reset_input_buffer()
        self._write_command(b"v")
        ready_text = self._wait_for_ready(timeout_s=12.0)
        if ready_text is None:
            self._close_serial()
            raise SerialBoardError(
                f"{self.port} @ {self.baud_rate}：发送 v 后未收到 $$$。\n"
                "请确认：① SW4/蓝牙适配器均为 921600；② 工具栏波特率选 921600；"
                "③ 板卡已上电且 COM 口未被其他程序占用。"
            )

        self._try_write(b"d", pause_s=0.2)
        response = self._read_response(timeout_s=2.5)
        if response.startswith("Failure"):
            self._close_serial()
            raise SerialBoardError(f"板卡 default 通道命令失败: {response.strip()}")
        self._initialized = True

    def release_session(self):
        if self._streaming:
            self.stop_stream()
        self._close_serial()
        self._initialized = False
        with self._lock:
            self._pending.clear()
            self._history.clear()
            self._partial = None
            self._samples_total = 0
            self._uart_packets_total = 0

    def start_stream(self, buffer_size=0, streamer_params=None):
        del buffer_size, streamer_params
        if not self._initialized or self._ser is None:
            raise SerialBoardError("板卡未就绪，请先连接。")
        if self._streaming:
            raise SerialBoardError("数据流已在运行。")
        with self._lock:
            self._pending.clear()
            self._partial = np.zeros(self.num_rows, dtype=np.float64)
            self._samples_total = 0
            self._uart_packets_total = 0
        self._write_command(b"b")
        self._keep_alive = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._streaming = True

    def stop_stream(self):
        if not self._streaming:
            return
        self._keep_alive = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._streaming = False
        self._try_write(b"s", pause_s=0.05)

    def config_board(self, cmd: str):
        if not self._initialized or self._ser is None:
            raise SerialBoardError("板卡未就绪。")
        payload = cmd.encode("ascii", errors="ignore")
        if self._streaming:
            self._ser.write(payload)
            return
        self._ser.write(payload)
        _ = self._read_response(timeout_s=1.5)

    def get_board_data(self) -> np.ndarray:
        with self._lock:
            if not self._pending:
                return np.zeros((self.num_rows, 0), dtype=np.float64)
            packages = list(self._pending)
            self._pending.clear()
        return np.column_stack(packages)

    def get_current_board_data(self, num_samples: int) -> np.ndarray:
        num_samples = max(0, int(num_samples))
        with self._lock:
            if num_samples == 0 or not self._history:
                return np.zeros((self.num_rows, 0), dtype=np.float64)
            take = min(num_samples, len(self._history))
            packages = list(self._history)[-take:]
        return np.column_stack(packages)

    def _push_package(self, package: np.ndarray):
        with self._lock:
            self._pending.append(package.copy())
            self._history.append(package.copy())
            self._samples_total += 1

    def _try_write(self, payload: bytes, pause_s: float = 0.0):
        if self._ser is None or not self._ser.is_open:
            return
        try:
            self._ser.write(payload)
            if pause_s > 0:
                time.sleep(pause_s)
        except serial.SerialException:
            pass

    def _write_command(self, payload: bytes):
        if self._ser is None:
            raise SerialBoardError("串口未打开。")
        self._ser.write(payload)

    def _read_response(self, timeout_s: float = 1.0) -> str:
        if self._ser is None:
            return ""
        deadline = time.monotonic() + timeout_s
        chunks: List[bytes] = []
        while time.monotonic() < deadline:
            waiting = self._ser.in_waiting
            if waiting:
                chunks.append(self._ser.read(waiting))
            else:
                time.sleep(0.02)
        return b"".join(chunks).decode("ascii", errors="ignore")

    def _wait_for_ready(self, timeout_s: float = 12.0) -> Optional[str]:
        if self._ser is None:
            return None
        deadline = time.monotonic() + timeout_s
        buf = ""
        while time.monotonic() < deadline:
            waiting = self._ser.in_waiting
            if waiting:
                chunk = self._ser.read(waiting).decode("ascii", errors="ignore")
                buf += chunk
                if "$$$" in buf:
                    return buf
            else:
                time.sleep(0.02)
        return None

    def _close_serial(self):
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            except serial.SerialException:
                pass
            self._ser = None

    def _read_loop(self):
        """921600 高速流：批量读入 + 滑动同步，避免逐字节读导致丢包。"""
        if self._ser is None:
            return
        stream_buf = bytearray()
        while self._keep_alive:
            try:
                waiting = self._ser.in_waiting
                if waiting:
                    chunk = self._ser.read(waiting)
                    if chunk:
                        stream_buf.extend(chunk)
                elif len(stream_buf) < 33:
                    time.sleep(0.001)
                    continue
            except serial.SerialException:
                time.sleep(0.005)
                continue

            while len(stream_buf) >= 33 and self._keep_alive:
                try:
                    start = stream_buf.index(START_BYTE)
                except ValueError:
                    stream_buf.clear()
                    break
                if start > 0:
                    del stream_buf[:start]
                if len(stream_buf) < 33:
                    break
                packet = bytes(stream_buf[:33])
                if packet[32] < END_BYTE_MIN or packet[32] > END_BYTE_MAX:
                    del stream_buf[0:1]
                    continue
                del stream_buf[:33]
                with self._lock:
                    self._uart_packets_total += 1
                self._handle_packet(packet)

    def _handle_packet(self, packet: bytes):
        if self._partial is None:
            self._partial = np.zeros(self.num_rows, dtype=np.float64)
        sample_id = packet[1]
        # BrainFlow cyton_daisy.cpp: even=daisy(先), odd=cyton(后) → odd 时提交整帧
        is_daisy_half = (sample_id % 2) == 0
        pkg = self._partial

        if is_daisy_half:
            for i in range(8):
                off = 2 + 3 * i
                val = EEG_SCALE * _cast_24bit_to_int32(packet[off: off + 3])
                row = self.eeg_channels[i + 8]
                pkg[row] = val
        else:
            for i in range(8):
                off = 2 + 3 * i
                val = EEG_SCALE * _cast_24bit_to_int32(packet[off: off + 3])
                row = self.eeg_channels[i]
                pkg[row] = val
            pkg[self.timestamp_channel] = time.time()
            self._push_package(pkg)
            self._partial = np.zeros(self.num_rows, dtype=np.float64)


def create_board(port: str, baud_rate: int, board_id: int):
    """115200 走 BrainFlow；921600 走 pyserial 直连。"""
    baud_rate = int(baud_rate)
    if baud_rate == 115200:
        from brainflow.board_shim import BrainFlowInputParams, BoardShim

        params = BrainFlowInputParams()
        params.serial_port = port
        return BoardShim(board_id, params), "BrainFlow"
    if baud_rate != 921600:
        raise SerialBoardError(f"不支持的波特率 {baud_rate}，请选择 115200 或 921600。")
    return PySerialCytonDaisyBoard(port, baud_rate, board_id), "pyserial"
