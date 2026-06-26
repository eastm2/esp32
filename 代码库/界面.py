# realtime_inference_gui_white_bg.py
import sys
import json
import serial
import joblib
import numpy as np
from collections import deque, Counter
from pathlib import Path
from scipy.signal import butter, filtfilt
import warnings

warnings.filterwarnings("ignore")

# PyQt5 GUI 相关导入
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import threading
import time

# ================== 配置 ==================
SERIAL_PORT = "COM6"
BAUDRATE = 115200
FS = 50
WINDOW_SIZE = 128
OVERLAP = 0.5
STEP = int(WINDOW_SIZE * (1 - OVERLAP))
INFERENCE_INTERVAL = STEP / FS
GRAVITY_CUTOFF = 0.3

CALIB_JSON = Path("calib_params.json")
MODEL_PATH = "har_rbf_svm_model.pkl"
BASELINE_PATH = "subject_baseline.joblib"

ACC_SCALE = 1 / 16384.0
GYRO_SCALE = 1 / 131.0
B_NORM = 46.83

SMOOTH_METHOD = "hmm"
MAJORITY_WINDOW = 5

# 活动标签映射
LABEL_MAP = {0: "静坐", 1: "站立", 2: "行走", 3: "跑步", 4: "上楼", 5: "下楼"}
ACTIVITY_ICONS = {
    0: "🪑", 1: "🧍", 2: "🚶", 3: "🏃", 4: "⬆️", 5: "⬇️"
}
ACTIVITY_COLORS = {
    0: "#4CAF50", 1: "#2196F3", 2: "#FF9800",
    3: "#F44336", 4: "#9C27B0", 5: "#607D8B"
}


# ================== GUI主窗口 ==================
class HAR_GUI(QMainWindow):
    # 自定义信号，用于线程安全地更新UI
    update_ui_signal = pyqtSignal(int, int, object)
    log_message_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_system()
        self.init_plots()

        # 连接信号和槽
        self.update_ui_signal.connect(self.update_ui_safe)
        self.log_message_signal.connect(self.log_message_safe)

        # 数据缓冲区
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.acc_buffer = deque(maxlen=500)
        self.gyro_buffer = deque(maxlen=500)
        self.time_buffer = deque(maxlen=500)
        self.start_time = time.time()

        # 推理相关
        self.last_inference = 0
        self.pred_history = deque(maxlen=MAJORITY_WINDOW)
        self.smoothed_label = None
        self.proba_history = deque(maxlen=100)
        self.is_running = False
        self.baseline_collected = False
        self.hardware_connected = False  # 新增：跟踪硬件连接状态

        # 定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(50)  # 20 FPS更新

    def init_ui(self):
        """初始化用户界面 - 白色背景版本"""
        self.setWindowTitle("人体活动识别系统 - 实时推理演示")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
                color: #212121;
            }
            QLabel {
                color: #212121;
                font-size: 14px;
            }
            QPushButton {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                padding: 10px 20px;
                border-radius: 5px;
                color: #212121;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
            QPushButton:pressed {
                background-color: #E0E0E0;
            }
            QPushButton:disabled {
                background-color: #FAFAFA;
                color: #9E9E9E;
                border: 1px solid #EEEEEE;
            }
            QGroupBox {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #212121;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #1976D2;
            }
            QTextEdit {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                color: #212121;
                font-family: "Consolas";
            }
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                text-align: center;
                background-color: #FAFAFA;
                color: #212121;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
            QStatusBar {
                background-color: #F5F5F5;
                color: #212121;
                border-top: 1px solid #E0E0E0;
            }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 左侧控制面板
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)

        # 右侧主显示区
        right_panel = self.create_main_display()
        main_layout.addWidget(right_panel, 3)

        # 底部状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪，等待连接硬件...")

    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title = QLabel("🏃 人体活动识别系统")
        title.setStyleSheet("""
            font-size: 22px; 
            font-weight: bold; 
            color: #1976D2; 
            padding: 10px;
            background-color: #E3F2FD;
            border-radius: 5px;
            border: 1px solid #BBDEFB;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 系统状态组
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)

        self.conn_status = QLabel("🔴 未连接")
        self.conn_status.setStyleSheet("""
            color: #D32F2F; 
            font-size: 16px;
            font-weight: bold;
            padding: 5px;
            background-color: #FFEBEE;
            border-radius: 3px;
        """)
        status_layout.addWidget(self.conn_status)

        self.model_status = QLabel("🤖 模型: RBF-SVM")
        self.model_status.setStyleSheet("padding: 3px;")
        status_layout.addWidget(self.model_status)

        self.smooth_status = QLabel(f"🔄 平滑: {SMOOTH_METHOD.upper()}")
        self.smooth_status.setStyleSheet("padding: 3px;")
        status_layout.addWidget(self.smooth_status)

        # 基准状态显示
        self.baseline_status = QLabel("📊 基准: 未采集")
        self.baseline_status.setStyleSheet("""
            color: #FF9800;
            font-size: 14px;
            font-weight: bold;
            padding: 5px;
            background-color: #FFF3E0;
            border-radius: 3px;
        """)
        status_layout.addWidget(self.baseline_status)

        layout.addWidget(status_group)

        # 控制按钮组
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(10)

        self.btn_connect = QPushButton("🔌 连接硬件")
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_connect.setStyleSheet("""
            QPushButton {
                background-color: #E3F2FD;
                border: 1px solid #2196F3;
                color: #1976D2;
            }
            QPushButton:hover {
                background-color: #BBDEFB;
            }
        """)
        control_layout.addWidget(self.btn_connect)

        self.btn_baseline = QPushButton("📊 采集基准")
        self.btn_baseline.clicked.connect(self.collect_baseline)
        self.btn_baseline.setEnabled(False)
        self.btn_baseline.setStyleSheet("""
            QPushButton {
                background-color: #E3F2FD;
                border: 1px solid #2196F3;
                color: #1976D2;
            }
            QPushButton:hover {
                background-color: #BBDEFB;
            }
            QPushButton:disabled {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                color: #9E9E9E;
            }
        """)
        control_layout.addWidget(self.btn_baseline)

        # 关键修改：先不禁用 btn_start，留到 init_system 里根据基准状态决定
        self.btn_start = QPushButton("▶️ 开始推理")
        self.btn_start.clicked.connect(self.toggle_inference)
        self.btn_start.setEnabled(False)  # 先禁用，等 init_system 决定
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #E8F5E9;
                border: 1px solid #4CAF50;
                color: #2E7D32;
            }
            QPushButton:hover {
                background-color: #C8E6C9;
            }
            QPushButton:disabled {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                color: #9E9E9E;
            }
        """)
        control_layout.addWidget(self.btn_start)

        self.btn_report = QPushButton("📝 生成报告")
        self.btn_report.clicked.connect(self.generate_report)
        self.btn_report.setEnabled(False)
        self.btn_report.setStyleSheet("""
            QPushButton {
                background-color: #FFF3E0;
                border: 1px solid #FF9800;
                color: #EF6C00;
            }
            QPushButton:hover {
                background-color: #FFE0B2;
            }
        """)
        control_layout.addWidget(self.btn_report)

        layout.addWidget(control_group)

        # 当前活动显示
        activity_group = QGroupBox("当前活动")
        activity_layout = QVBoxLayout(activity_group)
        activity_layout.setSpacing(10)
        activity_layout.setAlignment(Qt.AlignCenter)

        # 只保留活动文字标签
        self.activity_label = QLabel("未知")
        self.activity_label.setStyleSheet("""
            font-size: 36px; 
            font-weight: bold; 
            color: #1976D2;
            padding: 15px;
            background-color: #F5F5F5;
            border-radius: 10px;
            border: 2px solid #E0E0E0;
        """)
        self.activity_label.setAlignment(Qt.AlignCenter)
        activity_layout.addWidget(self.activity_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("置信度: %p%")
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                text-align: center;
                color: #212121;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        activity_layout.addWidget(self.confidence_bar)

        layout.addWidget(activity_group)

        # 移除传感器拓扑部分，直接添加伸缩项
        layout.addStretch()
        return panel

    def create_main_display(self):
        """创建右侧主显示区"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 顶部：活动概率分布
        # 顶部：活动指示块（原活动概率分布）
        indicator_group = QGroupBox("当前活动")
        indicator_layout = QGridLayout()
        indicator_layout.setSpacing(10)

        self.activity_blocks = {}
        for i in range(6):
            # 每个活动一个 QLabel 块
            block = QLabel(f"{ACTIVITY_ICONS[i]} {LABEL_MAP[i]}")
            block.setAlignment(Qt.AlignCenter)
            block.setMinimumHeight(50)
            block.setStyleSheet(f"""
                QLabel {{
                    background-color: #F0F0F0;
                    border: 2px solid #E0E0E0;
                    border-radius: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    color: #9E9E9E;
                    padding: 5px;
                }}
            """)
            # 按两行三列排列
            row, col = divmod(i, 3)
            indicator_layout.addWidget(block, row, col)
            self.activity_blocks[i] = block

        indicator_group.setLayout(indicator_layout)
        layout.addWidget(indicator_group)

        # 中部：传感器波形图
        plot_group = QGroupBox("传感器数据流")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setSpacing(10)

        # 加速度计波形
        acc_label = QLabel("加速度计 (g)")
        acc_label.setStyleSheet("font-weight: bold; color: #388E3C; padding: 5px;")
        plot_layout.addWidget(acc_label)

        self.acc_plot = pg.PlotWidget()
        self.acc_plot.setBackground('#FFFFFF')
        self.acc_plot.setYRange(-2, 2)
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_plot.getAxis('left').setTextPen('#212121')
        self.acc_plot.getAxis('bottom').setTextPen('#212121')
        self.acc_plot.getAxis('left').setPen(pg.mkPen('#BDBDBD'))
        self.acc_plot.getAxis('bottom').setPen(pg.mkPen('#BDBDBD'))
        self.acc_plot.getAxis('left').setGrid(192)
        self.acc_plot.getAxis('bottom').setGrid(192)
        plot_layout.addWidget(self.acc_plot)

        # 陀螺仪波形
        gyro_label = QLabel("陀螺仪 (°/s)")
        gyro_label.setStyleSheet("font-weight: bold; color: #C2185B; padding: 5px;")
        plot_layout.addWidget(gyro_label)

        self.gyro_plot = pg.PlotWidget()
        self.gyro_plot.setBackground('#FFFFFF')
        self.gyro_plot.setYRange(-250, 250)
        self.gyro_plot.showGrid(x=True, y=True, alpha=0.3)
        self.gyro_plot.getAxis('left').setTextPen('#212121')
        self.gyro_plot.getAxis('bottom').setTextPen('#212121')
        self.gyro_plot.getAxis('left').setPen(pg.mkPen('#BDBDBD'))
        self.gyro_plot.getAxis('bottom').setPen(pg.mkPen('#BDBDBD'))
        self.gyro_plot.getAxis('left').setGrid(192)
        self.gyro_plot.getAxis('bottom').setGrid(192)
        plot_layout.addWidget(self.gyro_plot)

        layout.addWidget(plot_group)

        # 底部：系统日志
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                color: #212121;
                font-family: "Consolas";
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        return panel

    def init_system(self):
        """初始化系统组件"""
        try:
            # 加载标定参数
            with open(CALIB_JSON, 'r') as f:
                self.calib = json.load(f)

            self.M_acc = np.array(self.calib["MPU6050_accelerometer"]["M_matrix"])
            self.b_acc = np.array(self.calib["MPU6050_accelerometer"]["bias_vector"])
            self.b_gyro = np.array(self.calib["MPU6050_gyroscope"]["bias_vector"])

            mag_params = self.calib["HMC5883L_magnetometer"]
            self.A_lsb = np.array(mag_params["transform_matrix_A_half"])
            self.c_lsb = np.array(mag_params["center_offset_c"])

            self.log_message_signal.emit("✅ 标定参数加载成功")
        except Exception as e:
            self.log_message_signal.emit(f"❌ 标定参数加载失败: {str(e)}")
            return

        try:
            # 加载模型
            self.model = joblib.load(MODEL_PATH)
            self.log_message_signal.emit("✅ SVM模型加载成功")
        except Exception as e:
            self.log_message_signal.emit(f"❌ 模型加载失败: {str(e)}")
            return

        # ===== 核心修改点：基准数据加载逻辑 =====
        # 默认状态
        self.baseline_collected = False
        self.pitch_base = 0.0
        self.roll_base = 0.0

        # 检查是否存在基准文件
        if Path(BASELINE_PATH).exists():
            try:
                self.baseline = joblib.load(BASELINE_PATH)
                self.pitch_base = self.baseline["pitch_base"]
                self.roll_base = self.baseline.get("roll_base", 0.0)
                self.baseline_collected = True  # 标记为已采集

                self.log_message_signal.emit(f"✅ 找到基准文件，已自动加载: 俯仰 {self.pitch_base:.2f}°")

                # UI状态：显示已加载
                self.baseline_status.setText("📊 基准: 已加载 (自动)")
                self.baseline_status.setStyleSheet("""
                    color: #388E3C;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 5px;
                    background-color: #E8F5E9;
                    border-radius: 3px;
                """)

                # 关键修复：启用重采基准按钮
                self.btn_baseline.setText("🔄 重采基准")
                self.btn_baseline.setEnabled(True)

                # 关键修复：如果有基准数据，即使硬件未连接，也要准备好开始推理按钮
                # 但硬件未连接时，按钮还是禁用的，因为需要硬件数据
                # 不过我们可以在连接硬件后立即启用
                self.log_message_signal.emit("ℹ️ 基准数据已就绪，连接硬件后可开始推理")

            except Exception as e:
                self.log_message_signal.emit(f"❌ 基准文件损坏，请重新采集: {str(e)}")
                self.baseline_status.setText("📊 基准: 文件损坏")
                self.baseline_status.setStyleSheet("""
                    color: #D32F2F;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 5px;
                    background-color: #FFEBEE;
                    border-radius: 3px;
                """)
                self.btn_baseline.setText("📊 采集基准")
                self.btn_baseline.setEnabled(True)
                self.btn_start.setEnabled(False)
        else:
            # 文件不存在
            self.log_message_signal.emit("ℹ️ 未找到基准文件，请先采集")
            self.baseline_status.setText("📊 基准: 未采集")
            self.baseline_status.setStyleSheet("""
                color: #FF9800;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #FFF3E0;
                border-radius: 3px;
            """)
            self.btn_baseline.setText("📊 采集基准")
            self.btn_baseline.setEnabled(True)
            self.btn_start.setEnabled(False)

        # 成本矩阵
        self.COST_MATRIX = np.ones((6, 6))
        np.fill_diagonal(self.COST_MATRIX, 0)
        self.COST_MATRIX[2, 5] = 5
        self.COST_MATRIX[4, 5] = 3
        self.COST_MATRIX[0, 1] = 3

        # 串口对象
        self.ser = None

    def init_plots(self):
        """初始化绘图曲线"""
        # 加速度计曲线
        colors = ['#F44336', '#4CAF50', '#2196F3']
        self.acc_curves = []
        for i, color in enumerate(colors):
            curve = self.acc_plot.plot(pen=pg.mkPen(color=color, width=2))
            self.acc_curves.append(curve)

        # 陀螺仪曲线
        colors = ['#FFC107', '#9C27B0', '#00BCD4']
        self.gyro_curves = []
        for i, color in enumerate(colors):
            curve = self.gyro_plot.plot(pen=pg.mkPen(color=color, width=2))
            self.gyro_curves.append(curve)

    def toggle_connection(self):
        """切换硬件连接状态"""
        if self.ser is None or not self.ser.is_open:
            try:
                self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
                self.conn_status.setText("🟢 已连接")
                self.conn_status.setStyleSheet("""
                    color: #388E3C; 
                    font-size: 16px;
                    font-weight: bold;
                    padding: 5px;
                    background-color: #E8F5E9;
                    border-radius: 3px;
                """)
                self.btn_connect.setText("🔌 断开连接")
                self.hardware_connected = True

                # 启用采集基准按钮（允许重采）
                self.btn_baseline.setEnabled(True)

                # 关键修复：如果基准已采集，立即启用开始推理按钮
                if self.baseline_collected:
                    self.btn_start.setEnabled(True)
                    self.log_message_signal.emit("✅ 硬件已连接，基准数据已就绪，可以开始推理")
                else:
                    self.btn_start.setEnabled(False)
                    self.log_message_signal.emit("⚠️ 硬件已连接，但请先采集基准数据")

                self.log_message_signal.emit(f"✅ 串口 {SERIAL_PORT} 已连接")
                self.status_bar.showMessage("硬件已连接，准备就绪")
            except Exception as e:
                self.log_message_signal.emit(f"❌ 连接失败: {str(e)}")
        else:
            self.ser.close()
            self.ser = None
            self.conn_status.setText("🔴 未连接")
            self.conn_status.setStyleSheet("""
                color: #D32F2F; 
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
                background-color: #FFEBEE;
                border-radius: 3px;
            """)
            self.btn_connect.setText("🔌 连接硬件")
            self.btn_baseline.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.is_running = False
            self.btn_start.setText("▶️ 开始推理")
            self.hardware_connected = False
            self.log_message_signal.emit("🔌 已断开连接")
            self.status_bar.showMessage("硬件已断开")

    def collect_baseline(self):
        """采集基准数据"""
        if self.ser is None or not self.ser.is_open:
            self.log_message_signal.emit("❌ 请先连接硬件")
            return

        self.log_message_signal.emit("📊 开始采集基准数据，请保持静坐30秒...")
        self.btn_baseline.setEnabled(False)
        self.btn_start.setEnabled(False)

        # 在新线程中采集基准
        thread = threading.Thread(target=self.baseline_thread)
        thread.daemon = True
        thread.start()

    def baseline_thread(self):
        """基准采集线程"""
        acc_raw_list = []
        start = time.time()

        while time.time() - start < 30:
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        parts = list(map(float, line.split(',')))
                        ax_raw, ay_raw, az_raw = parts[0], parts[1], parts[2]
                        acc_raw_list.append([ax_raw * ACC_SCALE, ay_raw * ACC_SCALE, az_raw * ACC_SCALE])
                    except:
                        pass

            # 更新进度
            elapsed = time.time() - start
            progress = int(elapsed / 30 * 100)
            self.status_bar.showMessage(f"采集基准中... {progress}%")

            time.sleep(0.01)

        if len(acc_raw_list) > 0:
            acc = np.array(acc_raw_list)
            gravity, _ = self.separate_gravity(acc, FS, GRAVITY_CUTOFF)
            pitch_raw, roll_raw, _ = self.calculate_raw_attitude(gravity)

            self.baseline = {
                "pitch_base": np.mean(pitch_raw),
                "roll_base": np.mean(roll_raw)
            }
            self.pitch_base = self.baseline["pitch_base"]
            self.roll_base = self.baseline["roll_base"]

            joblib.dump(self.baseline, BASELINE_PATH)
            self.baseline_collected = True

            self.log_message_signal.emit(f"✅ 基准采集完成: 俯仰 {self.pitch_base:.2f}°, 横滚 {self.roll_base:.2f}°")
            self.status_bar.showMessage("基准采集完成，可以开始推理")

            # 更新基准状态显示
            self.baseline_status.setText("📊 基准: 已采集")
            self.baseline_status.setStyleSheet("""
                color: #388E3C;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #E8F5E9;
                border-radius: 3px;
            """)

            # 更新按钮文本
            self.btn_baseline.setText("🔄 重采基准")

            # 启用开始推理按钮（硬件已连接）
            self.btn_start.setEnabled(True)
        else:
            self.log_message_signal.emit("❌ 基准采集失败: 未收到数据")
            self.status_bar.showMessage("基准采集失败")

            # 恢复按钮状态
            self.btn_baseline.setEnabled(True)

    def toggle_inference(self):
        """切换推理状态"""
        if not self.baseline_collected:
            self.log_message_signal.emit("❌ 请先采集基准数据")
            return

        if not self.is_running:
            self.is_running = True
            self.btn_start.setText("⏸️ 停止推理")
            self.btn_baseline.setEnabled(False)  # 推理时禁止重采
            self.log_message_signal.emit("🚀 开始实时推理...")
            self.status_bar.showMessage("实时推理运行中...")

            # 启动推理线程
            self.inference_thread = threading.Thread(target=self.inference_loop)
            self.inference_thread.daemon = True
            self.inference_thread.start()
        else:
            self.is_running = False
            self.btn_start.setText("▶️ 开始推理")
            self.btn_baseline.setEnabled(True)  # 停止推理后可以重采
            self.log_message_signal.emit("⏸️ 推理已停止")
            self.status_bar.showMessage("推理已停止")

    def inference_loop(self):
        """推理主循环"""
        last_inference = 0

        while self.is_running and self.ser and self.ser.is_open:
            # 读取串口数据
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, mx_raw, my_raw, mz_raw = \
                            map(float, line.split(','))

                        # 标定
                        ax, ay, az = self.calibrate_acc(ax_raw, ay_raw, az_raw)
                        gx, gy, gz = self.calibrate_gyro(gx_raw, gy_raw, gz_raw)
                        mx, my, mz = self.calibrate_mag(mx_raw, my_raw, mz_raw)

                        # 添加到缓冲区
                        self.buffer.append([ax, ay, az, gx, gy, gz, mx, my, mz])

                        # 添加到绘图缓冲区
                        current_time = time.time() - self.start_time
                        self.time_buffer.append(current_time)
                        self.acc_buffer.append([ax, ay, az])
                        self.gyro_buffer.append([gx, gy, gz])

                        # 定期推理
                        if len(self.buffer) >= WINDOW_SIZE and (time.time() - last_inference) >= INFERENCE_INTERVAL:
                            last_inference = time.time()
                            self.perform_inference()

                    except Exception as e:
                        pass

            time.sleep(0.001)  # 短暂休眠，避免CPU占用过高

    def perform_inference(self):
        """执行一次推理"""
        window = np.array(list(self.buffer))
        acc_raw = window[:, :3]
        gyro = window[:, 3:6]

        # 去重力
        gravity, acc_motion = self.separate_gravity(acc_raw, FS, GRAVITY_CUTOFF)

        # 提取特征
        feat = self.extract_sfs_features(acc_motion, gyro, gravity)
        x = np.array([feat])

        # 预测
        proba = self.model.predict_proba(x)[0]
        raw_pred = np.argmax(proba)

        # 平滑处理
        self.pred_history.append(raw_pred)
        if SMOOTH_METHOD == "majority" and len(self.pred_history) == MAJORITY_WINDOW:
            counter = Counter(self.pred_history)
            self.smoothed_label = counter.most_common(1)[0][0]
            self.smoothed_label = counter.most_common(1)[0][0]
        elif SMOOTH_METHOD == "hmm":
            if len(self.pred_history) >= 3:
                counter = Counter(list(self.pred_history)[-3:])
                self.smoothed_label = counter.most_common(1)[0][0]
            else:
                self.smoothed_label = raw_pred
        else:
            self.smoothed_label = raw_pred

        # 通过信号更新UI
        self.update_ui_signal.emit(raw_pred, self.smoothed_label, proba)

    @pyqtSlot(int, int, object)
    def update_ui_safe(self, raw_pred, smoothed_pred, proba):
        # 更新大标签
        self.activity_label.setText(LABEL_MAP[smoothed_pred])
        self.activity_label.setStyleSheet(
            f"font-size: 36px; font-weight: bold; color: {ACTIVITY_COLORS[smoothed_pred]}; "
            f"padding: 15px; background-color: #F5F5F5; border-radius: 10px; border: 2px solid #E0E0E0;")
        self.confidence_bar.setValue(int(proba[smoothed_pred] * 100))

        # 更新六个活动块
        for i in range(6):
            if i == smoothed_pred:
                self.activity_blocks[i].setStyleSheet(f"""
                    QLabel {{
                        background-color: {ACTIVITY_COLORS[i]}20;
                        border: 2px solid {ACTIVITY_COLORS[i]};
                        border-radius: 10px;
                        font-size: 14px;
                        font-weight: bold;
                        color: {ACTIVITY_COLORS[i]};
                        padding: 5px;
                    }}
                """)
            else:
                self.activity_blocks[i].setStyleSheet(f"""
                    QLabel {{
                        background-color: #F0F0F0;
                        border: 2px solid #E0E0E0;
                        border-radius: 10px;
                        font-size: 14px;
                        font-weight: bold;
                        color: #9E9E9E;
                        padding: 5px;
                    }}
                """)

        # 日志和状态栏
        timestamp = time.strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] 原始: {LABEL_MAP[raw_pred]}({proba[raw_pred]:.2f}) → 平滑: {LABEL_MAP[smoothed_pred]}"
        self.log_message_signal.emit(log_msg)
        self.status_bar.showMessage(f"当前活动: {LABEL_MAP[smoothed_pred]} | 置信度: {proba[smoothed_pred]:.2f}")


    @pyqtSlot(str)
    def log_message_safe(self, message):
        """线程安全的日志添加方法"""
        self.log_text.append(message)
        # 滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    def update_plots(self):
        """更新波形图"""
        if len(self.time_buffer) > 0:
            times = list(self.time_buffer)

            # 更新加速度计曲线
            if len(self.acc_buffer) > 0:
                acc_data = np.array(list(self.acc_buffer))
                for i in range(3):
                    if i < len(self.acc_curves):
                        self.acc_curves[i].setData(times, acc_data[:, i])

            # 更新陀螺仪曲线
            if len(self.gyro_buffer) > 0:
                gyro_data = np.array(list(self.gyro_buffer))
                for i in range(3):
                    if i < len(self.gyro_curves):
                        self.gyro_curves[i].setData(times, gyro_data[:, i])

    def generate_report(self):
        """生成报告"""
        self.log_message_signal.emit("📝 生成分析报告...")
        # 这里可以添加生成PDF报告的功能
        # 包括：活动统计、置信度分布、传感器数据质量等
        self.status_bar.showMessage("报告生成完成")

    # ================== 标定和信号处理函数 ==================
    def calibrate_acc(self, ax_raw, ay_raw, az_raw):
        ax = ax_raw * ACC_SCALE
        ay = ay_raw * ACC_SCALE
        az = az_raw * ACC_SCALE
        a_raw = np.array([ax - self.b_acc[0], ay - self.b_acc[1], az - self.b_acc[2]])
        a_cal = self.M_acc @ a_raw
        return a_cal[0], a_cal[1], a_cal[2]

    def calibrate_gyro(self, gx_raw, gy_raw, gz_raw):
        gx = gx_raw * GYRO_SCALE
        gy = gy_raw * GYRO_SCALE
        gz = gz_raw * GYRO_SCALE
        return gx - self.b_gyro[0], gy - self.b_gyro[1], gz - self.b_gyro[2]

    def calibrate_mag(self, mx_raw, my_raw, mz_raw):
        mag_lsb = np.array([mx_raw, my_raw, mz_raw])
        centered = mag_lsb - self.c_lsb
        mag_norm = self.A_lsb @ centered
        return (mag_norm * B_NORM).tolist()

    def separate_gravity(self, acc, fs, cutoff=0.3):
        nyq = 0.5 * fs
        normal_cutoff = min(cutoff / nyq, 0.99)
        b, a = butter(4, normal_cutoff, btype='low')
        gravity = np.zeros_like(acc)
        for i in range(3):
            gravity[:, i] = filtfilt(b, a, acc[:, i])
        return gravity, acc - gravity

    def calculate_raw_attitude(self, gravity):
        gx, gy, gz = gravity[:, 0], gravity[:, 1], gravity[:, 2]
        pitch = np.degrees(np.arctan2(-gx, np.sqrt(gy ** 2 + gz ** 2)))
        roll = np.degrees(np.arctan2(gy, gz))
        return pitch, roll, np.zeros_like(pitch)

    def extract_sfs_features(self, acc_motion, gyro, gravity_win):
        nyq = 0.5 * FS
        b_vib, a_vib = butter(2, [0.5 / nyq, 3.0 / nyq], btype='band')
        vib = filtfilt(b_vib, a_vib, acc_motion, axis=0)
        vib_energy = np.mean(np.sum(vib ** 2, axis=1))

        grav_norm = np.linalg.norm(gravity_win, axis=1) + 1e-8
        grav_ratio_y = np.mean(gravity_win[:, 1] / grav_norm)

        acc_mag = np.linalg.norm(acc_motion, axis=1)
        acc_mag_std = np.std(acc_mag)

        def band_energy_ratio(signal, fs, low, high):
            n = len(signal)
            freqs = np.fft.rfftfreq(n, d=1 / fs)
            psd = np.abs(np.fft.rfft(signal)) ** 2
            mask = (freqs >= low) & (freqs <= high)
            total = np.sum(psd[1:])
            band = np.sum(psd[mask])
            return band / (total + 1e-8)

        gyro_z_band = band_energy_ratio(gyro[:, 2], FS, 0.5, 2.0)

        pitch, roll, _ = self.calculate_raw_attitude(gravity_win)
        rel_pitch = -(pitch - self.pitch_base)
        pitch_rel_mean = np.mean(rel_pitch)

        acc_y_mean = np.mean(acc_motion[:, 1])

        return [vib_energy, grav_ratio_y, acc_mag_std, gyro_z_band, pitch_rel_mean, acc_y_mean]


# ================== 主程序入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")

    # 创建并显示主窗口
    gui = HAR_GUI()
    gui.show()

    # 运行应用程序
    sys.exit(app.exec_())