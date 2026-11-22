"""采样配置提示 + 采样率实测确认向导（软件不发 AT/~ 指令）。"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from .board_profile import (
    MANUAL_PROFILES,
    PROFILE_DEFAULT_125,
    BoardProfile,
)
from .rate_verify import (
    DEFAULT_MEASURE_DURATION_S,
    measure_stream_sample_rate,
    verify_profile,
)

# Step 1 说明文案（手册 §3 波特率、§4.5 采样率摘要）
INSTRUCTIONS_PROFILE_A = """
<h3>Profile A — 115200 · 帧 250 Hz（16 通道通道率 125 Hz）</h3>
<p>请在外部完成以下操作后再进入下一步：</p>
<ol>
<li>采集板 <b>SW4 拨码</b> 拨至 <b>115200</b> 侧，按手册完成上电与蓝牙适配器 AT 配置。</li>
<li>串口工具波特率设为 <b>115200</b>，发送 <code>AT</code> 应返回 <code>OK</code>。</li>
<li>发送 <code>~6</code> 将<b>帧采样率</b>设为 <b>250 Hz</b>（16 通道交错后每通道约 <b>125 Hz</b>），板卡应回复类似
<code>Success: Sample rate is 250Hz$$$</code>。</li>
<li>工具栏 <b>波特率</b> 选 <b>115200</b>，连接串口 → 开始采集 → 再打开「采样配置…」实测。</li>
</ol>
<p><i>向导实测的是 <b>16 通道通道采样率</b>（期望约 125 Hz），不是板卡回复里的 250 Hz。</i></p>
"""

INSTRUCTIONS_PROFILE_B = """
<h3>Profile B — 921600 · 帧 1000 Hz（16 通道通道率 500 Hz）</h3>
<p>请在外部完成以下操作后再进入下一步：</p>
<ol>
<li>采集板 <b>SW4 拨码</b> 拨至 <b>921600</b> 侧；配置时 D33 蓝灯应<b>闪烁</b>（未配对）。</li>
<li>先用 115200 发 <code>AT</code>，再发 <code>AT+BAUD=921600</code>，然后将 PC 串口改为 <b>921600</b> 并再次 <code>AT</code> 得 <code>OK</code>。</li>
<li>发送 <code>~4</code> 将<b>帧采样率</b>设为 <b>1000 Hz</b>（16 通道交错后每通道约 <b>500 Hz</b>）；仅 921600 可承载该数据流。</li>
<li>工具栏 <b>波特率</b> 选 <b>921600</b>，连接串口 → 开始采集 → 再打开「采样配置…」实测。</li>
</ol>
<p><b>注意：</b>921600 时程序用 pyserial 直连，不再走 BrainFlow 默认 115200。</p>
<p><i>向导实测的是 <b>16 通道通道采样率</b>（期望约 500 Hz），不是板卡回复里的 1000 Hz。</i></p>
"""


class SampleRateVerifyWorker(QtCore.QThread):
    """后台实测采样率，避免阻塞 UI。"""

    finished_ok = QtCore.pyqtSignal(float, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        board,
        profile: BoardProfile,
        eeg_row: int,
        duration_s: float = DEFAULT_MEASURE_DURATION_S,
        tolerance: float = 0.08,
        pause_ui_poll=None,
        resume_ui_poll=None,
        parent=None,
    ):
        super().__init__(parent)
        self.board = board
        self.profile = profile
        self.eeg_row = int(eeg_row)
        self.duration_s = float(duration_s)
        self.tolerance = float(tolerance)
        self.pause_ui_poll = pause_ui_poll
        self.resume_ui_poll = resume_ui_poll

    def run(self):
        try:
            measured = measure_stream_sample_rate(
                self.board,
                self.eeg_row,
                duration_s=self.duration_s,
                pause_ui_poll=self.pause_ui_poll,
                resume_ui_poll=self.resume_ui_poll,
            )
            ok, msg = verify_profile(self.profile, measured, tolerance=self.tolerance)
            if ok:
                self.finished_ok.emit(measured, msg)
            else:
                self.failed.emit(msg)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _IntroPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("采样配置说明")
        self.setSubTitle("请在外部完成波特率与采样率设置，本软件仅提供说明与后续实测确认。")

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        for title, html in (
            ("默认 · 帧 250 / 通道 125", INSTRUCTIONS_PROFILE_A),
            ("高速 · 帧 1000 / 通道 500", INSTRUCTIONS_PROFILE_B),
        ):
            browser = QtWidgets.QTextBrowser()
            browser.setOpenExternalLinks(True)
            browser.setHtml(html)
            tabs.addTab(browser, title)
        layout.addWidget(tabs)

        ack = QtWidgets.QCheckBox("我已在硬件/串口工具中完成对应配置")
        layout.addWidget(ack)
        self.registerField("hw_ack*", ack)


class _ChooseProfilePage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("选择目标档位")
        self.setSubTitle("请选择您刚才在外部完成的配置。")

        layout = QtWidgets.QVBoxLayout(self)
        self._group = QtWidgets.QButtonGroup(self)

        for idx, profile in enumerate(MANUAL_PROFILES):
            rb = QtWidgets.QRadioButton(
                f"{profile.display_label}\n"
                f"  期望波特率 {profile.baud_rate} · "
                f"通道 {profile.expected_sample_rate} Hz（帧 {profile.frame_sample_rate} Hz）"
            )
            rb.setProperty("profile_name", profile.name)
            if idx == 0:
                rb.setChecked(True)
            self._group.addButton(rb, idx)
            layout.addWidget(rb)

        layout.addStretch()

    def selected_profile(self) -> BoardProfile:
        btn = self._group.checkedButton()
        name = btn.property("profile_name") if btn else PROFILE_DEFAULT_125
        return BoardProfile.from_name(name)


class _VerifyPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("实测确认")
        self.setSubTitle("请先连接串口并开始采集，再点击「开始确认」。")

        self._board = None
        self._eeg_row = 0
        self._profile: Optional[BoardProfile] = None
        self._worker: Optional[SampleRateVerifyWorker] = None
        self._measured: Optional[float] = None
        self._result_msg = ""

        layout = QtWidgets.QVBoxLayout(self)
        self._status = QtWidgets.QLabel("等待连接并开始采集…")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._verify_btn = QtWidgets.QPushButton("开始确认")
        self._verify_btn.clicked.connect(self._start_verify)
        layout.addWidget(self._verify_btn)

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(200)
        layout.addWidget(self._log, 1)

        self.completeChanged.emit()

    def configure(
        self,
        board,
        eeg_row: int,
        profile: BoardProfile,
        is_connected: bool,
        is_streaming: bool,
    ):
        self._board = board
        self._eeg_row = int(eeg_row)
        self._profile = profile
        self._measured = None
        self._result_msg = ""
        if not is_connected:
            self._status.setText("未连接：请先在主界面连接串口。")
            self._verify_btn.setEnabled(False)
        elif not is_streaming:
            self._status.setText("未采集：请先在主界面点击「开始采集」。")
            self._verify_btn.setEnabled(False)
        else:
            self._status.setText(
                f"已连接且采集中。将实测约 {DEFAULT_MEASURE_DURATION_S:.0f} 秒，"
                f"期望通道 {profile.expected_sample_rate} Hz（帧 {profile.frame_sample_rate} Hz）。"
            )
            self._verify_btn.setEnabled(True)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._measured is not None

    def measured_hz(self) -> Optional[float]:
        return self._measured

    def result_message(self) -> str:
        return self._result_msg

    def _append_log(self, text: str):
        self._log.appendPlainText(text)

    def _start_verify(self):
        if self._board is None or self._profile is None:
            return
        self._verify_btn.setEnabled(False)
        self._append_log("开始实测采样率…")

        pause_cb = resume_cb = None
        main_win = self.wizard().parent()
        if main_win is not None and getattr(main_win, "is_streaming", False):
            timer = getattr(main_win, "timer", None)
            if timer is not None:
                interval_ms = timer.interval() or 40
                pause_cb = timer.stop
                resume_cb = lambda ms=interval_ms: timer.start(ms)

        self._worker = SampleRateVerifyWorker(
            self._board,
            self._profile,
            self._eeg_row,
            pause_ui_poll=pause_cb,
            resume_ui_poll=resume_cb,
            parent=self,
        )
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_ok(self, measured: float, msg: str):
        self._measured = measured
        self._result_msg = msg
        self._append_log(msg)
        self._status.setText(f"实测通过：{measured:.1f} Hz")
        self._verify_btn.setEnabled(True)
        self.completeChanged.emit()
        self.wizard().next()

    def _on_fail(self, err: str):
        self._append_log(err)
        self._status.setText(f"实测未通过：{err}")
        self._verify_btn.setEnabled(True)


class _ResultPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("确认结果")
        layout = QtWidgets.QVBoxLayout(self)
        self._label = QtWidgets.QLabel()
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

    def set_result(self, profile: BoardProfile, measured: float, detail: str):
        self._label.setText(
            f"<b>{profile.display_label}</b><br>"
            f"期望通道采样率：{profile.expected_sample_rate} Hz<br>"
            f"板卡帧采样率：{profile.frame_sample_rate} Hz<br>"
            f"实测通道采样率：{measured:.1f} Hz<br>"
            f"波特率：{profile.baud_rate}<br><br>"
            f"{detail}<br><br>"
            "点击「完成」将应用此配置并重建界面缓冲。"
        )


class HardwareSetupDialog(QtWidgets.QWizard):
    """四步向导：说明 → 选档 → 实测 → 结果。"""

    def __init__(
        self,
        parent=None,
        board=None,
        eeg_row: int = 0,
        is_connected: bool = False,
        is_streaming: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("采样配置确认")
        self.setMinimumSize(640, 520)

        self._board = board
        self._eeg_row = eeg_row
        self._is_connected = is_connected
        self._is_streaming = is_streaming
        self._confirmed_profile: Optional[BoardProfile] = None

        self._intro = _IntroPage(self)
        self._choose = _ChooseProfilePage(self)
        self._verify = _VerifyPage(self)
        self._result = _ResultPage(self)

        self.addPage(self._intro)
        self.addPage(self._choose)
        self.addPage(self._verify)
        self.addPage(self._result)

        self.setButtonText(QtWidgets.QWizard.FinishButton, "完成")
        self.currentIdChanged.connect(self._on_page_changed)

    def _on_page_changed(self, page_id: int):
        if self.page(page_id) is self._verify:
            profile = self._choose.selected_profile()
            self._verify.configure(
                self._board,
                self._eeg_row,
                profile,
                self._is_connected,
                self._is_streaming,
            )
        elif self.page(page_id) is self._result:
            profile = self._choose.selected_profile()
            measured = self._verify.measured_hz() or profile.expected_sample_rate
            self._result.set_result(profile, measured, self._verify.result_message())
            self._confirmed_profile = profile.with_measurement(measured)

    @staticmethod
    def run_wizard(
        parent=None,
        board=None,
        eeg_row: int = 0,
        is_connected: bool = False,
        is_streaming: bool = False,
    ) -> Optional[BoardProfile]:
        dlg = HardwareSetupDialog(
            parent=parent,
            board=board,
            eeg_row=eeg_row,
            is_connected=is_connected,
            is_streaming=is_streaming,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return dlg._confirmed_profile
