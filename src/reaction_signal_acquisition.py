import sys
import os
import time
import csv
import datetime
import re
import numpy as np
import scipy.stats as stats
import scipy.fft as fft
import scipy.signal as signal
import pywt
from hurst import compute_Hc

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QStatusBar, QMessageBox, QLineEdit, QSizePolicy)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import pyqtgraph as pg

# ================= Paths and driver configuration =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from python_bind.usb_server_binding import USBServerBinding, DeviceType
    from python_bind.daq122_binding import DAQ122Binding
    from python_bind.base_device_binding import (
        LockzhinerADCVoltage,
        LockzhinerADCSampleRate
    )
except ImportError as e:
    print(f"Warning: Failed to import the driver modules. Check the python_bind folder.\n{e}")

# DLL path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_NAME = "libdaq-4.1.2.dll" if os.name == 'nt' else "liblibdaq-3.10.4.so"
DLL_PATH = os.path.join(CURRENT_DIR, DLL_NAME)

# Data directory
DATA_DIR = os.path.join(CURRENT_DIR, "../data/reaction_measurements")

# ================= Global constants =================
FS = 1000.0
X_WINDOW_SIZE = 16384
UPDATE_STEP = 8192
ANALYSIS_WINDOW = 16384

Y_MIN = 0
Y_MAX = 0.8

SAMPLE_RATE_ENUM = LockzhinerADCSampleRate.ADCSampleRate_1_K
VOLTAGE_RANGE_ENUM = LockzhinerADCVoltage.ADCVoltage_5_V
READ_BATCH_SIZE = 100

MAIN_TITLE = '4-NPA Exploratory Experiment'

# Feature display order
FEATURE_KEYS = [
    "Mean", "Std", "Skewness", "Kurtosis",
    "Dom_Freq", "Spec_Energy", "Wave_Eng_D1", "Entropy",
    "Hurst", "Cross_Rate", "Peak_Dist", "Num_Peaks",
    "Num_Valleys", "Avg_Peak_H", "Avg_Valley_H", "Cross_Count"
]


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
    positive_freqs = fft_freqs[:N // 2]
    power_spectrum = np.abs(fft_vals[:N // 2]) / N
    idx_peak = np.argmax(power_spectrum)

    feats["Dom_Freq"] = positive_freqs[idx_peak]
    feats["Spec_Energy"] = power_spectrum[idx_peak]

    try:
        coeffs = pywt.wavedec(data, 'db4', level=3)
        energy_D1 = np.sum(np.square(coeffs[-1])) / len(coeffs[-1])
        feats["Wave_Eng_D1"] = energy_D1
    except:
        feats["Wave_Eng_D1"] = 0

    try:
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]
        feats["Entropy"] = -np.sum(hist * np.log2(hist))
    except:
        feats["Entropy"] = 0

    try:
        H, c, _ = compute_Hc(data, kind='change', simplified=False)
        feats["Hurst"] = H
    except:
        feats["Hurst"] = 0

    threshold = np.mean(data)
    centered_data = data - threshold
    zero_crossings = np.where(np.diff(np.sign(centered_data)))[0]
    count = len(zero_crossings)
    duration_seconds = len(data) / FS
    feats["Cross_Rate"] = count / duration_seconds if duration_seconds > 0 else 0
    feats["Cross_Count"] = count

    peaks, _ = signal.find_peaks(data, distance=20, prominence=0.1)
    num_peaks = len(peaks)
    avg_peak_height = np.mean(data[peaks]) if num_peaks > 0 else 0
    avg_peak_distance = np.mean(np.diff(peaks)) if num_peaks > 1 else 0

    valleys, _ = signal.find_peaks(-data, distance=20, prominence=0.1)
    num_valleys = len(valleys)
    avg_valley_height = np.mean(data[valleys]) if num_valleys > 0 else 0

    feats["Peak_Dist"] = avg_peak_distance
    feats["Num_Peaks"] = num_peaks
    feats["Num_Valleys"] = num_valleys
    feats["Avg_Peak_H"] = avg_peak_height
    feats["Avg_Valley_H"] = avg_valley_height

    return feats


# ================= Worker threads =================

class FeatureWorker(QThread):
    features_ready = pyqtSignal(dict, np.ndarray)

    def __init__(self):
        super().__init__()
        self.data_snapshot = None
        self.pending_calculation = False

    def set_data(self, data):
        self.data_snapshot = data
        self.pending_calculation = True
        if not self.isRunning():
            self.start()

    def run(self):
        if self.pending_calculation and self.data_snapshot is not None:
            try:
                feats = calculate_all_features(self.data_snapshot)
                self.features_ready.emit(feats, self.data_snapshot)
            except Exception as e:
                print(f"Feature processing error: {e}")
            finally:
                self.pending_calculation = False


class DAQWorker(QThread):
    data_received = pyqtSignal(np.ndarray)
    status_updated = pyqtSignal(str)
    rate_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.usb_server = None
        self.daq_device = None
        self._device_handle = None
        self._server_handle = None

    def run(self):
        try:
            self.status_updated.emit("Initializing USB service...")
            if not os.path.exists(DLL_PATH):
                raise FileNotFoundError(f"DLL file not found: {DLL_PATH}")

            self.usb_server = USBServerBinding(DLL_PATH)
            self._server_handle = self.usb_server.create_server(DeviceType.DeviceType_DAQ122)
            if not self._server_handle: raise Exception("Failed to create the USB server")

            self.usb_server.start_search(self._server_handle)
            timeout = 5
            start_time = time.time()
            found = False
            while time.time() - start_time < timeout:
                if self.usb_server.get_device_count(self._server_handle) > 0:
                    found = True
                    break
                time.sleep(0.1)
                if not self.is_running: return

            if not found: raise Exception("DAQ122 device not found")

            client_handle = self.usb_server.get_client_by_index(self._server_handle, 0)
            self.daq_device = DAQ122Binding(DLL_PATH)
            self._device_handle = self.daq_device.create_device()

            if not self.daq_device.use_backend(self._device_handle, client_handle): raise Exception("Failed to bind the backend")
            if not self.daq_device.initialize_device(self._device_handle): raise Exception("Failed to initialize the device")
            if not self.daq_device.connect_device(self._device_handle): raise Exception("Failed to connect to the device")

            self.daq_device.config_adc_sample_rate_and_voltage(self._device_handle, SAMPLE_RATE_ENUM,
                                                               VOLTAGE_RANGE_ENUM)

            self.daq_device.stop_collection(self._device_handle)
            self.daq_device.clear_data_buffer(self._device_handle)
            time.sleep(0.1)
            if not self.daq_device.start_collection(self._device_handle): raise Exception("Failed to start acquisition")

            self.status_updated.emit("Acquiring...")
            last_rate_time = time.time()
            total_points = 0

            while self.is_running:
                success, data = self.daq_device.try_read_data_batch(self._device_handle, 0, READ_BATCH_SIZE)
                if success and data:
                    arr = np.array(data)
                    self.data_received.emit(arr)
                    total_points += len(data)
                    now = time.time()
                    if now - last_rate_time >= 1.0:
                        self.rate_updated.emit(total_points / (now - last_rate_time))
                        total_points = 0
                        last_rate_time = now
                else:
                    time.sleep(0.002)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.cleanup()

    def stop(self):
        self.is_running = False
        self.wait()

    def cleanup(self):
        if self.daq_device and self._device_handle:
            try:
                self.daq_device.stop_collection(self._device_handle)
                self.daq_device.disconnect_device(self._device_handle)
                self.daq_device.delete_device(self._device_handle)
            except:
                pass
        if self.usb_server and self._server_handle:
            try:
                self.usb_server.delete_server(self._server_handle)
            except:
                pass
        self.status_updated.emit("Device disconnected")


# ================= Main window =================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"DAQ122 Acquisition, Analysis, and Recording - {MAIN_TITLE}")
        self.resize(1200, 700)

        # Buffers
        self.display_buffer = np.zeros(X_WINDOW_SIZE)
        self.feature_acc_buffer = []
        self.feature_window_buffer = np.zeros(ANALYSIS_WINDOW)
        self.has_filled_first_half = False

        self.feature_history = {key: [] for key in FEATURE_KEYS}
        self.feature_plot_refs = {}

        # Cached data
        self.cached_features = None
        self.cached_raw_data = None
        self.current_trial_dir = None

        # Timer state
        self.collection_start_time = 0
        self.ui_timer = QTimer()
        self.ui_timer.setInterval(200)
        self.ui_timer.timeout.connect(self.update_elapsed_time)

        # Threads
        self.worker = None
        self.feat_worker = FeatureWorker()
        self.feat_worker.features_ready.connect(self.on_features_calculated)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Feature plots
        feat_container = QWidget()
        feat_container.setFixedHeight(220)
        feat_layout = QGridLayout(feat_container)
        feat_layout.setSpacing(5)
        feat_layout.setContentsMargins(0, 0, 0, 0)

        for i, key in enumerate(FEATURE_KEYS):
            row = i // 8
            col = i % 8
            pw = pg.PlotWidget()
            pw.setBackground('k')
            pw.showGrid(x=False, y=True, alpha=0.5)
            pw.setTitle(key, color='w', size='8pt')
            pw.getAxis('bottom').setStyle(showValues=False)
            pw.getAxis('left').setStyle(tickTextWidth=30)
            pw.enableAutoRange(axis='x', enable=True)
            pw.enableAutoRange(axis='y', enable=True)
            curve = pw.plot(pen=pg.mkPen('c', width=1), symbol='o', symbolSize=3, symbolBrush='c')
            self.feature_plot_refs[key] = {'widget': pw, 'curve': curve}
            feat_layout.addWidget(pw, row, col)

        layout.addWidget(feat_container)

        # Main plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Voltage', units='V')
        self.plot_widget.setYRange(Y_MIN, Y_MAX, padding=0)
        self.plot_widget.setDownsampling(mode='peak')
        self.plot_widget.setClipToView(True)

        self.main_curve = self.plot_widget.plot(self.display_buffer, pen=pg.mkPen('y', width=2))
        layout.addWidget(self.plot_widget)

        # Controls
        control_layout = QHBoxLayout()

        self.txt_trial_name = QLineEdit()
        self.txt_trial_name.setPlaceholderText("Experiment name (default: notrial)")
        self.txt_trial_name.setFixedHeight(40)
        self.txt_trial_name.setStyleSheet("font-size: 14px; padding: 0 5px;")

        self.btn_auto_record = QPushButton("Continuous recording: ON")
        self.btn_auto_record.setCheckable(True)
        self.btn_auto_record.setChecked(True)
        self.btn_auto_record.setFixedHeight(40)
        self.btn_auto_record.clicked.connect(self.on_auto_record_toggled)
        self.update_record_btn_style()

        self.txt_note = QLineEdit()
        self.txt_note.setPlaceholderText("Note (default: nonote)")
        self.txt_note.setFixedHeight(40)
        self.txt_note.setStyleSheet("font-size: 14px; padding: 0 5px;")

        self.btn_save = QPushButton("Save features")
        self.btn_save.setFixedHeight(40)
        self.btn_save.setStyleSheet("font-size: 14px;")
        self.btn_save.clicked.connect(self.save_snapshot)

        self.btn_action = QPushButton("Connect and start acquisition")
        self.btn_action.setFixedHeight(40)
        self.btn_action.setStyleSheet("font-size: 14px;")
        self.btn_action.clicked.connect(self.toggle_collection)

        # Layout proportions: 3/16, 3/16, 3/16, 3/16, and 4/16.
        control_layout.addWidget(self.txt_trial_name, 3)  # 3/16
        control_layout.addWidget(self.btn_auto_record, 3)  # 3/16
        control_layout.addWidget(self.txt_note, 3)  # 3/16
        control_layout.addWidget(self.btn_save, 3)  # 3/16
        control_layout.addWidget(self.btn_action, 4)  # 4/16 (1/4)

        layout.addLayout(control_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.lbl_status = QLabel("Ready")

        self.lbl_timer = QLabel("00:00:00")
        self.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; margin-right: 15px;")

        self.lbl_rate = QLabel("0 SPS")
        self.lbl_rate.setStyleSheet("font-size: 14px;")
        self.lbl_rate.setFixedWidth(70)
        self.lbl_rate.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.status_bar.addWidget(self.lbl_status, 1)
        self.status_bar.addPermanentWidget(self.lbl_timer)
        self.status_bar.addPermanentWidget(self.lbl_rate)

    def update_record_btn_style(self):
        """Update the button style for the recording state."""
        if self.btn_auto_record.isChecked():
            self.btn_auto_record.setText("Continuous recording: ON")
            self.btn_auto_record.setStyleSheet(
                "background-color: #d0f0c0; color: black; font-size: 14px; border: 1px solid gray; border-radius: 4px;")
        else:
            self.btn_auto_record.setText("Continuous recording: OFF")
            self.btn_auto_record.setStyleSheet(
                "background-color: #f0d0d0; color: black; font-size: 14px; border: 1px solid gray; border-radius: 4px;")

    def on_auto_record_toggled(self, checked):
        """Handle changes to continuous recording."""
        self.update_record_btn_style()
        if not checked and self.worker and self.worker.isRunning() and self.current_trial_dir:
            try:
                filepath = os.path.join(self.current_trial_dir, "auto_record.csv")
                zeros = np.zeros((1000, 1))
                with open(filepath, 'a', newline='', encoding='utf-8-sig') as f:
                    np.savetxt(f, zeros, fmt='%d', delimiter=',')
                print(f"Inserted a pause marker (1,000 zero rows) into {filepath}")
            except Exception as e:
                print(f"Failed to insert the pause marker: {e}")

    def update_elapsed_time(self):
        """Update the elapsed-time display."""
        if self.collection_start_time > 0:
            elapsed = int(time.time() - self.collection_start_time)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.lbl_timer.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def toggle_collection(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.btn_action.setText("Connect and start acquisition")
            self.lbl_rate.setText("0 SPS")
            self.current_trial_dir = None

            self.ui_timer.stop()

            self.txt_trial_name.setEnabled(True)
        else:
            # Reset the UI and buffers.
            self.display_buffer = np.zeros(X_WINDOW_SIZE)
            self.main_curve.setData(self.display_buffer)
            self.feature_acc_buffer = []
            self.feature_window_buffer = np.zeros(ANALYSIS_WINDOW)
            self.has_filled_first_half = False
            self.cached_features = None
            self.cached_raw_data = None

            for key in FEATURE_KEYS:
                self.feature_history[key] = []
                self.feature_plot_refs[key]['curve'].setData([], [])

            self.lbl_timer.setText("00:00:00")
            self.collection_start_time = time.time()
            self.ui_timer.start()

            raw_trial_name = self.txt_trial_name.text().strip()
            safe_trial_name = re.sub(r'[\\/:*?"<>|]', '-', raw_trial_name)
            if not safe_trial_name:
                safe_trial_name = "notrial"

            now = datetime.datetime.now()
            date_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H%M%S")

            main_folder_name = f"{date_str}_{MAIN_TITLE}"
            trial_folder_name = f"{time_str}_{safe_trial_name}"

            self.current_trial_dir = os.path.join(DATA_DIR, main_folder_name, trial_folder_name)

            try:
                os.makedirs(self.current_trial_dir, exist_ok=True)
                self.lbl_status.setText(f"Folder created: {trial_folder_name}")
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Failed to create the experiment folder: {e}")
                self.ui_timer.stop()
                return

            self.txt_trial_name.setEnabled(False)

            self.worker = DAQWorker()
            self.worker.is_running = True
            self.worker.data_received.connect(self.process_data)
            self.worker.status_updated.connect(lambda s: self.lbl_status.setText(s))
            self.worker.rate_updated.connect(lambda r: self.lbl_rate.setText(f"{r:.0f} SPS"))
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()
            self.btn_action.setText("Stop acquisition")

    def process_data(self, new_data):
        num = len(new_data)
        if num == 0: return

        # Continuous recording
        if self.btn_auto_record.isChecked() and self.current_trial_dir:
            try:
                filepath = os.path.join(self.current_trial_dir, "auto_record.csv")
                with open(filepath, 'a', newline='', encoding='utf-8-sig') as f:
                    np.savetxt(f, new_data, fmt='%.6f', delimiter=',')
            except Exception as e:
                print(f"Auto record error: {e}")

        # Update the display.
        self.display_buffer = np.roll(self.display_buffer, -num)
        self.display_buffer[-num:] = new_data
        self.main_curve.setData(self.display_buffer)

        self.feature_acc_buffer.extend(new_data)

        while len(self.feature_acc_buffer) >= UPDATE_STEP:
            chunk = np.array(self.feature_acc_buffer[:UPDATE_STEP])
            self.feature_acc_buffer = self.feature_acc_buffer[UPDATE_STEP:]

            if not self.has_filled_first_half:
                self.feature_window_buffer[:UPDATE_STEP] = chunk
                self.has_filled_first_half = True
            else:
                self.feature_window_buffer[:UPDATE_STEP] = self.feature_window_buffer[UPDATE_STEP:]
                self.feature_window_buffer[UPDATE_STEP:] = chunk
                self.feat_worker.set_data(self.feature_window_buffer.copy())

    def on_features_calculated(self, feats, raw_data):
        if not feats: return
        self.cached_features = feats
        self.cached_raw_data = raw_data

        for key in FEATURE_KEYS:
            val = feats.get(key, 0)
            self.feature_history[key].append(val)
            if len(self.feature_history[key]) > 200:
                self.feature_history[key].pop(0)
            y_data = self.feature_history[key]
            self.feature_plot_refs[key]['curve'].setData(np.arange(len(y_data)), y_data)

    def save_snapshot(self):
        if self.cached_features is None or self.cached_raw_data is None:
            QMessageBox.warning(self, "Notice", "No completed feature calculation is available to save.")
            return

        if not self.current_trial_dir:
            QMessageBox.warning(self, "Notice", "No active experiment folder was found. Connect the device first.")
            return

        raw_note = self.txt_note.text()
        raw_note = raw_note.strip()
        note_suffix = re.sub(r'[\\/:*?"<>|]', '-', raw_note)
        if not note_suffix:
            note_suffix = "nonote"

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%H%M%S")
        tsdata_title = f"tsdata_{timestamp_str}_{note_suffix}"

        try:
            feat_table_path = os.path.join(self.current_trial_dir, "features_table.csv")
            file_exists = os.path.exists(feat_table_path)

            with open(feat_table_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ["tsdata_title"] + FEATURE_KEYS
                    writer.writerow(header)
                row_data = [tsdata_title] + [self.cached_features.get(k, 0) for k in FEATURE_KEYS]
                writer.writerow(row_data)

            ts_filename = f"{tsdata_title}.csv"
            ts_path = os.path.join(self.current_trial_dir, ts_filename)
            np.savetxt(ts_path, self.cached_raw_data, fmt='%.6f', delimiter=',', encoding='utf-8-sig')

            self.lbl_status.setText(f"Saved: {tsdata_title}")
            self.flash_save_success()

        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def flash_save_success(self):
        original_style = self.btn_save.styleSheet()
        self.btn_save.setStyleSheet("background-color: #ccffcc; font-size: 14px;")
        self.btn_save.setText("Saved!")

        from PyQt6.QtCore import QTimer
        def restore():
            try:
                self.btn_save.setStyleSheet(original_style)
                self.btn_save.setText("Save features")
            except:
                pass

        QTimer.singleShot(1000, restore)

    def handle_error(self, err_msg):
        QMessageBox.critical(self, "Error", err_msg)
        self.toggle_collection()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
