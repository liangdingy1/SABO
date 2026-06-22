import sys
import os
import time
import csv
import datetime
import numpy as np
import scipy.stats as stats
import scipy.fft as fft
import scipy.signal as signal
import pywt
from hurst import compute_Hc

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QMessageBox, QComboBox, QGroupBox, QProgressBar,
    QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal
import pyqtgraph as pg

# ================= Experimental constants =================
TUBE_VOL_1M = 0.7854           # Volume of 1 m tubing (mL)
SYRINGE_VOL_LIMIT = 19.0       # Maximum safe syringe volume (mL)
STABILIZE_FACTOR = 1.5         # Steady-state waiting factor

# ================= Serial ports and pump mapping =================
AQ_PORT = "COM4"
ORG_PORT = "COM6"
AQ_SLAVE_ID = 1
ORG_SLAVE_ID = 1

# ================= DAQ driver imports =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from python_bind.usb_server_binding import USBServerBinding, DeviceType
    from python_bind.daq122_binding import DAQ122Binding
    from python_bind.base_device_binding import (
        LockzhinerADCVoltage,
        LockzhinerADCSampleRate
    )
    DAQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import the DAQ driver. Simulated data mode will be used.\n{e}")
    DAQ_AVAILABLE = False

# DLL path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_NAME = "libdaq-4.1.2.dll" if os.name == 'nt' else "liblibdaq-3.10.4.so"
DLL_PATH = os.path.join(CURRENT_DIR, DLL_NAME)
DATA_DIR = os.path.join(CURRENT_DIR, "../data/Data")
os.makedirs(DATA_DIR, exist_ok=True)

# Global constants
FS = 1000.0
X_WINDOW_SIZE = 16384
UPDATE_STEP = 8192
ANALYSIS_WINDOW = 16384
Y_MIN = 0
Y_MAX = 0.8
SAMPLE_RATE_ENUM = LockzhinerADCSampleRate.ADCSampleRate_1_K if DAQ_AVAILABLE else None
VOLTAGE_RANGE_ENUM = LockzhinerADCVoltage.ADCVoltage_5_V if DAQ_AVAILABLE else None
READ_BATCH_SIZE = 100

FEATURE_KEYS = [
    "Mean", "Std", "Skewness", "Kurtosis",
    "Dom_Freq", "Spec_Energy", "Wave_Eng_D1", "Entropy",
    "Hurst", "Cross_Rate", "Peak_Dist", "Num_Peaks",
    "Num_Valleys", "Avg_Peak_H", "Avg_Valley_H", "Cross_Count"
]

# ================= Experimental conditions (Q_total, Q_aq, Q_org) =================
EXPERIMENT_CONDITIONS = []
EXPERIMENT_CONDITIONS.extend([(0.2, 0.06, 0.14), (0.2, 0.08, 0.12), (0.2, 0.10, 0.10), (0.2, 0.12, 0.08), (0.2, 0.14, 0.06)])
EXPERIMENT_CONDITIONS.extend([(0.4, 0.12, 0.28), (0.4, 0.16, 0.24), (0.4, 0.20, 0.20), (0.4, 0.24, 0.16), (0.4, 0.28, 0.12)])
EXPERIMENT_CONDITIONS.extend([(0.6, 0.18, 0.42), (0.6, 0.24, 0.36), (0.6, 0.30, 0.30), (0.6, 0.36, 0.24), (0.6, 0.42, 0.18)])
EXPERIMENT_CONDITIONS.extend([(0.8, 0.24, 0.56), (0.8, 0.32, 0.48), (0.8, 0.40, 0.40), (0.8, 0.48, 0.32), (0.8, 0.56, 0.24)])
EXPERIMENT_CONDITIONS.extend([(1.0, 0.30, 0.70), (1.0, 0.40, 0.60), (1.0, 0.50, 0.50), (1.0, 0.60, 0.40), (1.0, 0.70, 0.30)])

# ================= Modbus pump control =================
try:
    from pymodbus.client import ModbusSerialClient
    from pymodbus.payload import BinaryPayloadBuilder
    from pymodbus.constants import Endian
    from pymodbus.exceptions import ModbusIOException
    MODBUS_AVAILABLE = True
except Exception as e:
    print(f"Warning: pymodbus is unavailable; automatic pump control is disabled.\n{e}")
    MODBUS_AVAILABLE = False

REG_WORK_TYPE    = 1029  # 0: volume mode; 1: flow-rate mode
REG_RUN_MODE     = 1004  # 1: infusion
REG_INFUSE_VOL   = 1005  # float uL
REG_INFUSE_FLOW  = 1025  # float uL/min
REG_STARTSTOP    = 1019  # 0 stop / 1 run / 2 pause

BYTEORDER = Endian.BIG
WORDORDER = Endian.BIG


def fmt_phi(phi: float) -> str:
    """Format 0.50 as 0.5, 1.00 as 1, and retain values such as 0.33."""
    s = f"{phi:.2f}"
    s = s.rstrip('0').rstrip('.')
    return s


class ISPLabPump:
    """Control one pump on one serial port, using slave ID 1 by default."""
    def __init__(self, port: str, slave: int, name: str):
        if not MODBUS_AVAILABLE:
            raise RuntimeError("pymodbus is not installed; pump control is unavailable.")
        self.port = port
        self.slave = slave
        self.name = name
        self.client = ModbusSerialClient(
            port=port,
            baudrate=9600,
            bytesize=8,
            parity="E",
            stopbits=1,
            timeout=0.6
        )

    def connect(self) -> bool:
        return self.client.connect()

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def _write_u16(self, addr: int, val: int, retries: int = 5) -> None:
        last = None
        for k in range(retries):
            try:
                rr = self.client.write_register(address=addr, value=val, slave=self.slave)
                if rr.isError():
                    last = rr
                    time.sleep(0.08 * (k + 1))
                    continue
                return
            except (ModbusIOException, OSError) as e:
                last = e
                time.sleep(0.10 * (k + 1))
        raise RuntimeError(f"{self.name}: write_u16 failed addr={addr} val={val} last={last}")

    def _write_f32(self, addr: int, val: float, retries: int = 5) -> None:
        last = None
        for k in range(retries):
            try:
                b = BinaryPayloadBuilder(byteorder=BYTEORDER, wordorder=WORDORDER)
                b.add_32bit_float(float(val))
                regs = b.to_registers()
                rr = self.client.write_registers(address=addr, values=regs, slave=self.slave)
                if rr.isError():
                    last = rr
                    time.sleep(0.08 * (k + 1))
                    continue
                return
            except (ModbusIOException, OSError) as e:
                last = e
                time.sleep(0.10 * (k + 1))
        raise RuntimeError(f"{self.name}: write_f32 failed addr={addr} val={val} last={last}")

    def stop(self):  self._write_u16(REG_STARTSTOP, 0)
    def start(self): self._write_u16(REG_STARTSTOP, 1)
    def pause(self): self._write_u16(REG_STARTSTOP, 2)

    def set_infuse_volume_mode_and_start(self, flow_mlmin: float, volume_ml: float):
        """
        STOP -> work_type=0 -> run_mode=1 -> infuse_vol -> infuse_flow -> START
        """
        self.stop()
        time.sleep(0.05)
        self._write_u16(REG_WORK_TYPE, 0)      # volume mode
        self._write_u16(REG_RUN_MODE, 1)       # infuse
        self._write_f32(REG_INFUSE_VOL, volume_ml * 1000.0)      # mL -> uL
        self._write_f32(REG_INFUSE_FLOW, flow_mlmin * 1000.0)    # mL/min -> uL/min
        time.sleep(0.05)
        self.start()


# ================= Feature calculation =================
def calculate_all_features(data):
    if len(data) != ANALYSIS_WINDOW:
        return {}

    feats = {}
    feats["Mean"] = np.mean(data)
    feats["Std"] = np.std(data)
    feats["Skewness"] = stats.skew(data)
    feats["Kurtosis"] = stats.kurtosis(data)

    data_centered = data - np.mean(data)
    N = len(data_centered)
    fft_vals = fft.fft(data_centered)
    fft_freqs = fft.fftfreq(N, 1 / FS)
    power_spectrum = np.abs(fft_vals[:N // 2]) / N
    idx_peak = np.argmax(power_spectrum)
    feats["Dom_Freq"] = fft_freqs[idx_peak]
    feats["Spec_Energy"] = power_spectrum[idx_peak]

    try:
        coeffs = pywt.wavedec(data, 'db4', level=3)
        feats["Wave_Eng_D1"] = np.sum(np.square(coeffs[-1])) / len(coeffs[-1])
    except:
        feats["Wave_Eng_D1"] = 0

    try:
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]
        feats["Entropy"] = -np.sum(hist * np.log2(hist))
    except:
        feats["Entropy"] = 0

    try:
        H, _, _ = compute_Hc(data, kind='change', simplified=False)
        feats["Hurst"] = H
    except:
        feats["Hurst"] = 0

    zero_crossings = np.where(np.diff(np.sign(data - np.mean(data))))[0]
    feats["Cross_Rate"] = len(zero_crossings) / (len(data) / FS)
    feats["Cross_Count"] = len(zero_crossings)

    peaks, _ = signal.find_peaks(data, distance=20, prominence=0.1)
    valleys, _ = signal.find_peaks(-data, distance=20, prominence=0.1)
    feats["Num_Peaks"] = len(peaks)
    feats["Num_Valleys"] = len(valleys)
    feats["Peak_Dist"] = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
    feats["Avg_Peak_H"] = np.mean(data[peaks]) if len(peaks) > 0 else 0
    feats["Avg_Valley_H"] = np.mean(data[valleys]) if len(valleys) > 0 else 0

    return feats


class DAQWorker(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.daq_device = None
        self._device_handle = None
        self.usb_server = None
        self._server_handle = None

    def stop(self):
        self.is_running = False
        self.wait(800)

    def run(self):
        if not DAQ_AVAILABLE:
            # Simulated data mode
            t = 0
            while self.is_running:
                chunk = 0.4 + 0.2 * np.sin(2 * np.pi * 2 * np.linspace(t, t + 0.1, 100)) + np.random.normal(0, 0.02, 100)
                self.data_received.emit(chunk)
                t += 0.1
                time.sleep(0.1)
            return

        try:
            self.usb_server = USBServerBinding(DLL_PATH)
            self._server_handle = self.usb_server.create_server(DeviceType.DeviceType_DAQ122)
            self.usb_server.start_search(self._server_handle)

            start = time.time()
            while time.time() - start < 3:
                if self.usb_server.get_device_count(self._server_handle) > 0:
                    break
                time.sleep(0.1)

            client_handle = self.usb_server.get_client_by_index(self._server_handle, 0)
            self.daq_device = DAQ122Binding(DLL_PATH)
            self._device_handle = self.daq_device.create_device()
            self.daq_device.use_backend(self._device_handle, client_handle)
            self.daq_device.initialize_device(self._device_handle)
            self.daq_device.connect_device(self._device_handle)
            self.daq_device.config_adc_sample_rate_and_voltage(self._device_handle, SAMPLE_RATE_ENUM, VOLTAGE_RANGE_ENUM)
            self.daq_device.start_collection(self._device_handle)

            while self.is_running:
                success, data = self.daq_device.try_read_data_batch(self._device_handle, 0, READ_BATCH_SIZE)
                if success and data:
                    self.data_received.emit(np.array(data))
                else:
                    time.sleep(0.002)

        except Exception as e:
            print(f"DAQ Error: {e}")
        finally:
            if self.daq_device:
                try:
                    self.daq_device.stop_collection(self._device_handle)
                    self.daq_device.disconnect_device(self._device_handle)
                except:
                    pass


# ================= Automated experiment thread =================
class ExperimentController(QThread):
    sig_status = pyqtSignal(str)
    sig_progress = pyqtSignal(int, int)
    sig_req_capture3 = pyqtSignal(int, float, float)  # (cond_index(1-based), q_aq, q_org)
    sig_pump_usage = pyqtSignal(float, float)         # (used_aq, used_org)
    sig_wait_left = pyqtSignal(int)                   # Remaining stabilization time
    sig_finished = pyqtSignal()
    sig_ask_refill = pyqtSignal(str, str)             # (title, message)
    sig_ask_start = pyqtSignal(str, str)              # (title, message)

    def __init__(self, tube_length_m: int, start_cond_index: int, aq_pump: ISPLabPump, org_pump: ISPLabPump):
        super().__init__()
        self.tube_vol = TUBE_VOL_1M * tube_length_m

        self.is_running = True

        self.user_confirmed = False
        self.user_refilled = False

        self.aq_pump = aq_pump
        self.org_pump = org_pump

        self.vol_used_aq = 0.0
        self.vol_used_org = 0.0

        self.pending_saves = 0

        # Resume from this one-based condition index.
        self.start_cond_index = max(1, int(start_cond_index))

    def user_confirm_action(self):
        self.user_confirmed = True

    def user_refill_action(self):
        self.user_refilled = True

    def notify_one_save_done(self):
        if self.pending_saves > 0:
            self.pending_saves -= 1

    def reset_volume_counter(self):
        self.vol_used_aq = 0.0
        self.vol_used_org = 0.0
        self.sig_pump_usage.emit(0.0, 0.0)

    def stop_experiment(self):
        self.is_running = False
        try:
            self.aq_pump.stop()
            self.org_pump.stop()
        except Exception:
            pass
        self.wait(800)

    def _predict_step_usage(self, q_aq, q_org, wait_time_sec):
        # Three overlapping windows add 16,384 + 8,192 + 8,192 points.
        capture_time = (16384 + 8192 + 8192) / FS
        total_time = wait_time_sec + capture_time + 6.0  # Allow time for saving and thread overhead.
        return (q_aq * total_time / 60.0, q_org * total_time / 60.0)

    def _maybe_refill(self, next_usage_aq, next_usage_org, idx1_based, total_steps):
        if (self.vol_used_aq + next_usage_aq > SYRINGE_VOL_LIMIT) or (self.vol_used_org + next_usage_org > SYRINGE_VOL_LIMIT):
            try:
                self.aq_pump.stop()
                self.org_pump.stop()
            except Exception:
                pass

            self.user_refilled = False
            title = "Refill Required"
            msg = (
                f"Experiment progress: {idx1_based}/{total_steps}\n\n"
                "The next condition is expected to exceed the 19 mL safety threshold.\n"
                "Replace both the Aq and Org 20 mL syringes after each 19 mL of use.\n\n"
                "After refilling, click \"Refilled; continue\". The volume counters will reset "
                "and the automated experiment will continue."
            )
            self.sig_ask_refill.emit(title, msg)

            while (not self.user_refilled) and self.is_running:
                self.msleep(100)

            if not self.is_running:
                return False

            self.reset_volume_counter()
        return True

    def run(self):
        total_steps = len(EXPERIMENT_CONDITIONS)
        start_idx0 = min(max(self.start_cond_index - 1, 0), total_steps - 1)

        self.sig_status.emit("Connecting syringe pumps (COM4=Aq, COM6=Org)...")
        if not self.aq_pump.connect():
            self.sig_status.emit("Failed to connect the Aq pump (COM4)")
            self.sig_finished.emit()
            return
        if not self.org_pump.connect():
            self.sig_status.emit("Failed to connect the Org pump (COM6)")
            self.aq_pump.close()
            self.sig_finished.emit()
            return

        self.user_confirmed = False
        self.sig_ask_start.emit(
            "Pre-run Check",
            "Confirm the following:\n"
            "1) Both pumps are on their main screens, not a settings screen.\n"
            "2) Communication is enabled.\n"
            "3) COM4 is Aq and COM6 is Org.\n\n"
            f"This run will start at condition {start_idx0+1} of {total_steps}.\n"
            "Click \"Start automated experiment\" to continue."
        )
        while (not self.user_confirmed) and self.is_running:
            self.msleep(100)
        if not self.is_running:
            try:
                self.aq_pump.close()
                self.org_pump.close()
            except Exception:
                pass
            self.sig_finished.emit()
            return

        try:
            for idx0 in range(start_idx0, total_steps):
                if not self.is_running:
                    break

                idx1 = idx0 + 1
                q_tot, q_aq, q_org = EXPERIMENT_CONDITIONS[idx0]

                self.sig_progress.emit(idx1, total_steps)

                residence_time_min = (self.tube_vol / q_tot)
                wait_time_sec = residence_time_min * STABILIZE_FACTOR * 60.0

                next_usage_aq, next_usage_org = self._predict_step_usage(q_aq, q_org, wait_time_sec)
                if not self._maybe_refill(next_usage_aq, next_usage_org, idx1, total_steps):
                    break

                try:
                    self.sig_status.emit(f"[{idx1}/{total_steps}] Setting flow rates and starting pumps: Aq={q_aq:.2f}, Org={q_org:.2f} mL/min")
                    self.aq_pump.set_infuse_volume_mode_and_start(flow_mlmin=q_aq, volume_ml=SYRINGE_VOL_LIMIT)
                    self.org_pump.set_infuse_volume_mode_and_start(flow_mlmin=q_org, volume_ml=SYRINGE_VOL_LIMIT)
                except Exception as e:
                    self.sig_status.emit(f"Failed to configure or start the pumps: {e}")
                    break

                remaining = int(round(wait_time_sec))
                while remaining > 0 and self.is_running:
                    self.sig_wait_left.emit(remaining)
                    self.sig_status.emit(f"[{idx1}/{total_steps}] Stabilizing... {remaining} s remaining | Aq={q_aq:.2f}, Org={q_org:.2f}")
                    self.msleep(1000)
                    remaining -= 1

                    dt_min = 1.0 / 60.0
                    self.vol_used_aq += q_aq * dt_min
                    self.vol_used_org += q_org * dt_min
                    self.sig_pump_usage.emit(self.vol_used_aq, self.vol_used_org)

                if not self.is_running:
                    break

                self.sig_status.emit(f"[{idx1}/{total_steps}] Steady state reached. Capturing three overlapping windows (16384 -> +8192 -> +8192)...")
                self.pending_saves = 3
                self.sig_req_capture3.emit(idx1, q_aq, q_org)

                wait_s = 0.0
                while self.pending_saves > 0 and self.is_running:
                    self.msleep(100)
                    wait_s += 0.1
                    if wait_s >= 1.0:
                        self.vol_used_aq += q_aq * (wait_s / 60.0)
                        self.vol_used_org += q_org * (wait_s / 60.0)
                        self.sig_pump_usage.emit(self.vol_used_aq, self.vol_used_org)
                        wait_s = 0.0

                self.msleep(300)

            self.sig_status.emit("All experiments are complete. The pumps have been stopped automatically.")

        finally:
            try:
                self.aq_pump.stop()
                self.org_pump.stop()
            except Exception:
                pass
            try:
                self.aq_pump.close()
                self.org_pump.close()
            except Exception:
                pass

            self.sig_finished.emit()


# ================= Main window =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automated Microfluidic Flow-Regime Mapping")
        self.resize(1120, 760)

        # Display buffer
        self.display_buffer = np.zeros(X_WINDOW_SIZE)

        # Sliding window for live monitoring only
        self.feature_acc_buffer = []
        self.feature_window_buffer = np.zeros(ANALYSIS_WINDOW)
        self.has_filled_first_half = False

        # Output paths
        self.current_main_folder = None
        self.features_file_path = None

        # Three-window gated acquisition
        self.capture_active = False
        self.capture_queue = []
        self.capture_needed = 0
        self.capture_rep = 1
        self.capture_prev_window = None
        self.capture_q_aq = 0.0
        self.capture_q_org = 0.0
        self.capture_cond_idx1 = 1

        # Rate limit for capture progress updates
        self._last_capture_status_ts = 0.0
        self._capture_hint = ""

        self.init_ui()

        self.daq_worker = DAQWorker()
        self.daq_worker.data_received.connect(self.process_daq_data)
        self.daq_worker.is_running = True
        self.daq_worker.start()

        self.exp_thread = None

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        top_group = QGroupBox("Automated Experiment Settings")
        top_layout = QHBoxLayout()

        top_layout.addWidget(QLabel("Tube length:"))
        self.combo_tube = QComboBox()
        self.combo_tube.addItems(["1m", "2m", "4m"])
        top_layout.addWidget(self.combo_tube)

        top_layout.addWidget(QLabel("Starting condition (1-25):"))
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(1)
        self.spin_start.setMaximum(len(EXPERIMENT_CONDITIONS))
        self.spin_start.setValue(1)
        self.spin_start.setToolTip("Resume after an interruption by selecting the first condition to run.")
        top_layout.addWidget(self.spin_start)

        self.btn_start = QPushButton("Start automated experiment")
        self.btn_start.setStyleSheet("background-color: #ccffcc; padding: 6px; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_experiment)
        top_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop experiment and pumps")
        self.btn_stop.setStyleSheet("background-color: #ffdddd; padding: 6px; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_experiment)
        top_layout.addWidget(self.btn_stop)

        self.btn_reset_vol = QPushButton("Reset volume after refill")
        self.btn_reset_vol.clicked.connect(self.reset_volumes)
        top_layout.addWidget(self.btn_reset_vol)

        top_group.setLayout(top_layout)
        layout.addWidget(top_group)

        mon_layout = QGridLayout()
        self.lbl_status = QLabel("Select the tube length and click Start")
        self.lbl_status.setStyleSheet("color: blue; font-size: 14px;")

        self.progress_bar = QProgressBar()
        self.lbl_wait = QLabel("Stabilization remaining: - s")
        self.lbl_vol_aq = QLabel("Aq cumulative volume: 0.00 mL")
        self.lbl_vol_org = QLabel("Org cumulative volume: 0.00 mL")
        self.lbl_capture = QLabel("Capture status: -")
        self.lbl_capture.setStyleSheet("color: #444444;")

        mon_layout.addWidget(self.lbl_status, 0, 0, 1, 3)
        mon_layout.addWidget(self.progress_bar, 0, 3, 1, 1)
        mon_layout.addWidget(self.lbl_wait, 1, 0, 1, 2)
        mon_layout.addWidget(self.lbl_capture, 1, 2, 1, 2)
        mon_layout.addWidget(self.lbl_vol_aq, 2, 0, 1, 2)
        mon_layout.addWidget(self.lbl_vol_org, 2, 2, 1, 2)

        layout.addLayout(mon_layout)

        self.plot_widget = pg.PlotWidget(title="Live Waveform")
        self.plot_widget.setYRange(Y_MIN, Y_MAX)
        self.curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)

    def start_experiment(self):
        if not MODBUS_AVAILABLE:
            QMessageBox.critical(self, "Missing dependency", "pymodbus is not installed, so automatic pump control is unavailable. Run: pip install pymodbus")
            return

        tube_str = self.combo_tube.currentText()
        tube_len = int(tube_str.replace('m', ''))
        start_cond = int(self.spin_start.value())

        # Reuse the same folder for the same date and tube length.
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        folder_name = f"{date_str}_{tube_str} flow regime mapping"
        self.current_main_folder = os.path.join(DATA_DIR, folder_name)
        os.makedirs(self.current_main_folder, exist_ok=True)

        # Use a v2 file if an existing feature table has a different header.
        desired_header = ["Replicate_ID", "Q_aq", "Q_org", "phi_aq", "Q_total"] + FEATURE_KEYS
        base_path = os.path.join(self.current_main_folder, "features_table.csv")
        path_to_use = base_path

        if os.path.exists(base_path):
            try:
                with open(base_path, 'r', newline='') as f:
                    first = f.readline().strip()
                old_cols = [c.strip() for c in first.split(",")] if first else []
                if old_cols != desired_header:
                    path_to_use = os.path.join(self.current_main_folder, "features_table_v2.csv")
            except Exception:
                path_to_use = os.path.join(self.current_main_folder, "features_table_v2.csv")

        self.features_file_path = path_to_use
        if not os.path.exists(self.features_file_path):
            with open(self.features_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(desired_header)

        # Lock controls during acquisition.
        self.combo_tube.setEnabled(False)
        self.spin_start.setEnabled(False)
        self.btn_start.setEnabled(False)

        aq_pump = ISPLabPump(AQ_PORT, AQ_SLAVE_ID, "Aq Pump(COM4)")
        org_pump = ISPLabPump(ORG_PORT, ORG_SLAVE_ID, "Org Pump(COM6)")

        self.exp_thread = ExperimentController(tube_len, start_cond, aq_pump, org_pump)
        self.exp_thread.sig_status.connect(self.lbl_status.setText)
        self.exp_thread.sig_progress.connect(self.update_progress)
        self.exp_thread.sig_pump_usage.connect(self.update_volumes)
        self.exp_thread.sig_wait_left.connect(self.update_wait_left)
        self.exp_thread.sig_req_capture3.connect(self.start_capture3_session)
        self.exp_thread.sig_finished.connect(self.on_finished)
        self.exp_thread.sig_ask_refill.connect(self.show_refill_dialog)
        self.exp_thread.sig_ask_start.connect(self.show_start_dialog)
        self.exp_thread.start()

    def stop_experiment(self):
        if self.exp_thread:
            self.exp_thread.stop_experiment()
            self.lbl_status.setText("Stopped; pump shutdown requested")
        self._stop_capture_session()

    def _stop_capture_session(self):
        self.capture_active = False
        self.capture_queue = []
        self.capture_prev_window = None
        self.capture_needed = 0
        self.capture_rep = 1
        self.lbl_capture.setText("Capture status: -")

    def show_start_dialog(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Information)
        btn_ok = msg_box.addButton("Start automated experiment", QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg_box.exec()
        if msg_box.clickedButton() == btn_ok:
            self.exp_thread.user_confirm_action()
        else:
            self.stop_experiment()

    def show_refill_dialog(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Warning)

        btn_ok = msg_box.addButton("Refilled; continue", QMessageBox.ButtonRole.AcceptRole)
        btn_stop = msg_box.addButton("Stop experiment", QMessageBox.ButtonRole.RejectRole)
        msg_box.exec()

        if msg_box.clickedButton() == btn_ok:
            self.exp_thread.user_refill_action()
        else:
            self.stop_experiment()

    def reset_volumes(self):
        if self.exp_thread:
            self.exp_thread.reset_volume_counter()
            QMessageBox.information(self, "Reset", "The volume counters have been reset.")

    def update_progress(self, curr, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)

    def update_wait_left(self, sec_left: int):
        self.lbl_wait.setText(f"Stabilization remaining: {sec_left} s")

    def update_volumes(self, aq, org):
        self.lbl_vol_aq.setText(f"Aq cumulative volume: {aq:.2f} mL (threshold: {SYRINGE_VOL_LIMIT:.0f} mL)")
        self.lbl_vol_org.setText(f"Org cumulative volume: {org:.2f} mL (threshold: {SYRINGE_VOL_LIMIT:.0f} mL)")

    # Overlapping-window capture session
    def start_capture3_session(self, cond_index_1based: int, q_aq: float, q_org: float):
        """
        Start immediately after stabilization:
        rep1 accumulates 16,384 points;
        rep2 adds 8,192 points and saves the overlapping window;
        rep3 repeats the 8,192-point update.
        """
        if self.capture_active:
            self._stop_capture_session()

        self.capture_active = True
        self.capture_queue = []
        self.capture_needed = 16384
        self.capture_rep = 1
        self.capture_prev_window = None
        self.capture_q_aq = float(q_aq)
        self.capture_q_org = float(q_org)
        self.capture_cond_idx1 = int(cond_index_1based)

        self._last_capture_status_ts = 0.0
        self._capture_hint = ""
        self._update_capture_status(force=True)

    def _make_filename(self, q_aq: float, q_org: float, rep: int) -> str:
        total = q_aq + q_org
        phi = (q_aq / total) if total > 0 else 0.0

        time_str = datetime.datetime.now().strftime("%H%M%S")
        filename = f"{time_str}_aq{q_aq:.2f}-org{q_org:.2f}-phi{fmt_phi(phi)}-total{total:.1f}-{rep}.csv"
        return filename

    def _save_one_window(self, window: np.ndarray, rep: int, q_aq: float, q_org: float):
        """Save the waveform CSV and append its feature row."""
        total = q_aq + q_org
        phi = (q_aq / total) if total > 0 else 0.0

        filename = self._make_filename(q_aq, q_org, rep)
        np.savetxt(os.path.join(self.current_main_folder, filename), window, fmt='%.6f', delimiter=',')

        feats = calculate_all_features(window)

        # Replicate_ID is the complete CSV filename.
        with open(self.features_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [filename, q_aq, q_org, phi, total] + [feats.get(k, 0) for k in FEATURE_KEYS]
            writer.writerow(row)

        print(f"Saved: {filename}")

    def _update_capture_status(self, force: bool = False):
        """Display remaining points and time for each replicate."""
        now = time.time()
        if (not force) and (now - self._last_capture_status_ts < 0.25):
            return
        self._last_capture_status_ts = now

        if not self.capture_active:
            self.lbl_capture.setText("Capture status: -")
            return

        remain_pts = max(self.capture_needed - len(self.capture_queue), 0)
        remain_sec = remain_pts / FS
        hint = f"Capture status: condition {self.capture_cond_idx1}/25 | rep {self.capture_rep}/3 | {remain_pts} points remaining (~{remain_sec:.2f} s)"
        # Avoid redundant UI updates.
        if hint != self._capture_hint:
            self._capture_hint = hint
            self.lbl_capture.setText(hint)

    def _process_capture_queue(self):
        """Save replicates when the capture queue reaches each point target."""
        if not self.capture_active:
            return

        while self.capture_active and (len(self.capture_queue) >= self.capture_needed):
            take = self.capture_queue[:self.capture_needed]
            self.capture_queue = self.capture_queue[self.capture_needed:]
            new_chunk = np.array(take, dtype=float)

            if self.capture_rep == 1:
                window = new_chunk
                self.capture_prev_window = window.copy()
                self._save_one_window(window, rep=1, q_aq=self.capture_q_aq, q_org=self.capture_q_org)
                if self.exp_thread:
                    self.exp_thread.notify_one_save_done()

                self.capture_rep = 2
                self.capture_needed = 8192
                self._update_capture_status(force=True)

            elif self.capture_rep == 2:
                prev = self.capture_prev_window
                window = np.concatenate([prev[8192:], new_chunk])
                self.capture_prev_window = window.copy()
                self._save_one_window(window, rep=2, q_aq=self.capture_q_aq, q_org=self.capture_q_org)
                if self.exp_thread:
                    self.exp_thread.notify_one_save_done()

                self.capture_rep = 3
                self.capture_needed = 8192
                self._update_capture_status(force=True)

            elif self.capture_rep == 3:
                prev = self.capture_prev_window
                window = np.concatenate([prev[8192:], new_chunk])
                self.capture_prev_window = window.copy()
                self._save_one_window(window, rep=3, q_aq=self.capture_q_aq, q_org=self.capture_q_org)
                if self.exp_thread:
                    self.exp_thread.notify_one_save_done()

                self._stop_capture_session()
                break

    def process_daq_data(self, data):
        # Live display
        self.display_buffer = np.roll(self.display_buffer, -len(data))
        self.display_buffer[-len(data):] = data
        self.curve.setData(self.display_buffer)

        # Sliding window for monitoring only
        self.feature_acc_buffer.extend(data)
        while len(self.feature_acc_buffer) >= UPDATE_STEP:
            chunk = np.array(self.feature_acc_buffer[:UPDATE_STEP])
            self.feature_acc_buffer = self.feature_acc_buffer[UPDATE_STEP:]

            if not self.has_filled_first_half:
                self.feature_window_buffer[:UPDATE_STEP] = chunk
                self.has_filled_first_half = True
            else:
                self.feature_window_buffer[:UPDATE_STEP] = self.feature_window_buffer[UPDATE_STEP:]
                self.feature_window_buffer[UPDATE_STEP:] = chunk

        # Gated overlapping-window acquisition
        if self.capture_active:
            self.capture_queue.extend(np.asarray(data, dtype=float).tolist())
            self._update_capture_status(force=False)
            self._process_capture_queue()
            self._update_capture_status(force=False)

    def on_finished(self):
        QMessageBox.information(self, "Experiment Complete", "All conditions have been acquired.\nThe pumps were stopped automatically.")
        self.btn_start.setEnabled(True)
        self.combo_tube.setEnabled(True)
        self.spin_start.setEnabled(True)
        self.exp_thread = None
        self.lbl_wait.setText("Stabilization remaining: - s")
        self._stop_capture_session()

    def closeEvent(self, event):
        try:
            if self.daq_worker:
                self.daq_worker.stop()
        except Exception:
            pass
        try:
            if self.exp_thread:
                self.exp_thread.stop_experiment()
        except Exception:
            pass
        self._stop_capture_session()
        event.accept()

    # ==== UI widgets init after methods =====
    def _post_init_plot(self):
        pass


def build_window():
    win = MainWindow()
    return win


def init_plot_widget(win: MainWindow, layout: QVBoxLayout):
    win.plot_widget = pg.PlotWidget(title="Live Waveform")
    win.plot_widget.setYRange(Y_MIN, Y_MAX)
    win.curve = win.plot_widget.plot(pen='y')
    layout.addWidget(win.plot_widget)


# Patch: attach plot in init_ui (kept simple)
# We re-define MainWindow.init_ui at runtime is ugly; instead we integrate above.
# The plot is already created in init_ui earlier, we just need to ensure those attributes exist.
# (In this single-file script, they do.)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    # plot_widget is created inside init_ui
    win.show()
    sys.exit(app.exec())
