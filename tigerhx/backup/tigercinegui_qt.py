import sys
import glob
import numpy as np
from os.path import basename
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QTextEdit, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QSizeGrip, QSpacerItem, QSizePolicy, QSlider, QProgressBar
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QPoint

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Background, QColor(45, 45, 45))
        self.setPalette(palette)

        self.setFixedHeight(40)
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.layout.addSpacerItem(QSpacerItem(40, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.title = QLabel("TigerHx GUI")
        self.title.setStyleSheet("color: white; font: bold 20px; margin-left: 10px;")
        self.layout.addWidget(self.title)

        self.minimize_button = QPushButton("-")
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.setStyleSheet("background-color: #333; color: white; border: none;")
        self.minimize_button.clicked.connect(parent.showMinimized)
        self.layout.addWidget(self.minimize_button)

        self.maximize_button = QPushButton("□")
        self.maximize_button.setFixedSize(30, 30)
        self.maximize_button.setStyleSheet("background-color: #333; color: white; border: none;")
        self.maximize_button.clicked.connect(self.toggle_maximize_restore)
        self.layout.addWidget(self.maximize_button)

        self.close_button = QPushButton("×")
        self.close_button.setFixedSize(30, 30)
        self.close_button.setStyleSheet("background-color: #333; color: white; border: none;")
        self.close_button.clicked.connect(parent.close)
        self.layout.addWidget(self.close_button)

        self.setLayout(self.layout)
        self.parent = parent

    def toggle_maximize_restore(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def mousePressEvent(self, event):
        self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.old_pos)
        self.parent.move(self.parent.x() + delta.x(), self.parent.y() + delta.y())
        self.old_pos = event.globalPos()

class TigerHxGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setGeometry(100, 100, 1400, 900)
        
        # Custom title bar
        self.title_bar = CustomTitleBar(self)
        self.setMenuWidget(self.title_bar)

        # Set the palette
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))  # Background color
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))  # Text color
        palette.setColor(QPalette.Base, QColor(45, 45, 45))  # Text edit background color
        palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))  # Alternate background color
        palette.setColor(QPalette.ToolTipBase, Qt.white)  # Tooltip background color
        palette.setColor(QPalette.ToolTipText, Qt.white)  # Tooltip text color
        palette.setColor(QPalette.Text, QColor(220, 220, 220))  # Text color
        palette.setColor(QPalette.Button, QColor(50, 50, 50))  # Button color
        palette.setColor(QPalette.ButtonText, Qt.white)  # Button text color
        palette.setColor(QPalette.BrightText, Qt.red)  # Bright text color
        palette.setColor(QPalette.Link, QColor(42, 130, 218))  # Hyperlink color
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))  # Highlight color
        palette.setColor(QPalette.HighlightedText, Qt.white)  # Highlighted text color

        self.setPalette(palette)

        # Set the style for scrollbar, combobox, button, text edit etc.
        self.setStyleSheet("""
            QScrollBar:vertical {
                background-color: #333;
                width: 15px;
                margin: 15px 3px 15px 3px;
            }
            QScrollBar::handle:vertical {
                background-color: #666;
                min-height: 5px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background-color: #444;
                height: 14px;
                subcontrol-origin: margin;
                subcontrol-position: top;
                border: 1px solid #2e2e2e;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                border: 2px solid #2e2e2e;
                width: 3px;
                height: 3px;
                background: #888;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QComboBox, QPushButton, QListWidget, QTextEdit {
                background-color: #333;
                border: 1px solid #555;
                color: white;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                border: 1px solid #555;
                color: white;
                selection-background-color: #555;
            }
            QSlider::groove:vertical {
                border: 1px solid #bbb;
                background: #333;
                width: 8px;
                border-radius: 4px;
            }
            QSlider::handle:vertical {
                background: #666;
                border: 1px solid #5c5c5c;
                height: 20px;
                margin: -2px 0;
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: #555;
            }
            QProgressBar {
                text-align: center;
                color: white;
                background-color: #333;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #666;
                width: 20px;
                margin: 1px;
            }
            QLabel {
                color: white;
            }
        """)

        # Main layout
        main_layout = QHBoxLayout()

        # Left layout
        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(500)  # Set the width of the left layout

        # Selection area
        selection_layout = QVBoxLayout()

        self.combo_box = QComboBox()
        self.combo_box.setFont(QFont("Arial", 11))
        self.load_models()

        self.label = QLabel("Please select a segmentation model")
        self.label.setFont(QFont("Arial", 11))

        self.go_button = QPushButton("Go")
        self.go_button.setFont(QFont("Arial", 11))
        self.go_button.clicked.connect(self.on_go_button_clicked)

        selection_layout.addWidget(self.combo_box)
        selection_layout.addWidget(self.label)
        selection_layout.addWidget(self.go_button)

        left_layout.addLayout(selection_layout)

        # Text Edit area
        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont("Arial", 11))
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText("TigerHX GUI started...")
        
        left_layout.addWidget(self.text_edit)

        # Figure type and colormap area
        figure_layout = QHBoxLayout()
        self.figure_label = QLabel("Figure type")
        self.figure_label.setFont(QFont("Arial", 11))
        self.figure_combo = QComboBox()
        self.figure_combo.setFont(QFont("Arial", 11))
        self.figure_combo.addItems(["Seg_crop", "Option2", "Option3"])

        self.colormap_label = QLabel("Colormap")
        self.colormap_label.setFont(QFont("Arial", 11))
        self.colormap_combo = QComboBox()
        self.colormap_combo.setFont(QFont("Arial", 11))
        self.colormap_combo.addItems(["vivid", "Option2", "Option3"])

        figure_layout.addWidget(self.figure_label)
        figure_layout.addWidget(self.figure_combo)
        figure_layout.addWidget(self.colormap_label)
        figure_layout.addWidget(self.colormap_combo)

        left_layout.addLayout(figure_layout)

        # List area
        self.list_widget = QListWidget()
        self.list_widget.setFont(QFont("Arial", 11))
        self.load_mat_files()

        left_layout.addWidget(self.list_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(50)
        left_layout.addWidget(self.progress_bar)

        # Image display area on the right
        self.image_label = QLabel()
        self.image_label.setFixedSize(600, 800)  # Set the size of the image display area
        self.display_random_image()

        # Right layout with vertical slider
        right_layout = QVBoxLayout()
        self.vertical_slider = QSlider(Qt.Vertical)
        self.vertical_slider.setMinimum(0)
        self.vertical_slider.setMaximum(100)
        self.vertical_slider.setValue(50)
        right_layout.addWidget(self.vertical_slider)

        # Add left, center (image), and right layouts to the main layout
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(right_layout)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Size grip for resizing the window
        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet("background-color: #333;")

    def load_models(self):
        model_files = glob.glob('models/*.onnx')
        for model_file in model_files:
            self.combo_box.addItem(basename(model_file))

    def load_mat_files(self):
        mat_files = glob.glob('output/*.mat')
        for mat_file in mat_files:
            self.list_widget.addItem(basename(mat_file))

    def display_random_image(self):
        # Generate random data
        data = np.random.rand(800, 600) * 255
        data = data.astype(np.uint8)

        # Convert numpy array to QImage
        height, width = data.shape
        q_image = QImage(data.data, width, height, width, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def on_go_button_clicked(self):
        self.text_edit.append("Go button clicked!")
        self.display_random_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TigerHxGUI()
    window.show()
    sys.exit(app.exec_())
