from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QBrush
from PyQt5.QtCore import Qt
import sys
import os


class ImageRecommenderApp(QWidget):
    def __init__(self):
        super().__init__()

        self.logo_path = "C:/Users/meist/Image-Recommender/logo.png"
        self.background_path = "C:/Users/meist/Image-Recommender/App_background.jpg"

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Recommender")
        self.setWindowIcon(QIcon(self.logo_path))

        # Set window size to available screen size (excluding taskbar)
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen_geometry)

        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 20, 40, 30)

        # Logo + Title
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_pixmap = QPixmap(self.logo_path)
        logo_label.setPixmap(logo_pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        title_label = QLabel("Image Recommender")
        title_label.setFont(QFont("Arial", 48, QFont.Bold))
        title_label.setStyleSheet("color: #e60028;")
        title_label.setAlignment(Qt.AlignCenter)

        header_layout.addStretch()
        header_layout.addWidget(logo_label)
        header_layout.addSpacing(20)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # Input + Browse
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Enter image path...")
        self.input_path.setFixedWidth(400)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_image)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(browse_button)

        # Preview selected image
        self.image_preview = QLabel()
        self.image_preview.setFixedSize(300, 300)
        self.image_preview.setStyleSheet("border: 2px solid #ccc; background-color: white;")
        self.image_preview.setAlignment(Qt.AlignCenter)

        input_preview_layout = QHBoxLayout()
        input_preview_layout.addStretch()
        input_preview_layout.addLayout(input_layout)
        input_preview_layout.addSpacing(40)
        input_preview_layout.addWidget(self.image_preview)
        input_preview_layout.addStretch()

        # Results label
        self.results_label = QLabel("Top 3 Results:")
        self.results_label.setFont(QFont("Arial", 18))
        self.results_label.setAlignment(Qt.AlignCenter)

        # Combine layouts
        main_layout.addLayout(header_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(input_preview_layout)
        main_layout.addWidget(self.results_label)
        main_layout.addStretch()

        self.setLayout(main_layout)
        self.setAutoFillBackground(True)
        self.update_background()

    def update_background(self):
        if os.path.exists(self.background_path):
            palette = self.palette()
            pixmap = QPixmap(self.background_path).scaled(
                self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            palette.setBrush(QPalette.Window, QBrush(pixmap))
            self.setPalette(palette)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_background()

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.input_path.setText(file_path)
            self.display_input_image(file_path)

    def display_input_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_preview.setPixmap(
                pixmap.scaled(self.image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageRecommenderApp()
    window.showMaximized()
    sys.exit(app.exec_())
