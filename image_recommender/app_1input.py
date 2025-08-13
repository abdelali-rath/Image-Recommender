import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from image_recommender.pipeline.search_pipeline import combined_similarity_search



class SearchThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, query_path, index_path, mapping_path, k_clip=20, top_k_result=5):
        super().__init__()
        self.query_path = query_path
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.k_clip = k_clip
        self.top_k_result = top_k_result

    def run(self):
        try:
            results = combined_similarity_search(
                self.query_path,
                clip_index_path=self.index_path,
                clip_mapping_path=self.mapping_path,
                k_clip=self.k_clip,
                top_k_result=self.top_k_result
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.logo_path = os.path.join(os.path.dirname(__file__), '..', 'logo.png')
        self.background_path = os.path.join(os.path.dirname(__file__), '..', 'App_background.jpg')
        self.CLIP_INDEX_PATH = os.path.abspath("data/out/clip_index.ann")
        self.CLIP_MAPPING_PATH = os.path.abspath("data/out/index_to_id.json")

        # background
        self.bg_pixmap = QPixmap(self.background_path) if os.path.exists(self.background_path) else QPixmap()

        self.search_thread = None
        self.result_widgets = []  # (img_label, score_label)

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Image Recommender")
        if os.path.exists(self.logo_path):
            self.setWindowIcon(QIcon(self.logo_path))

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(36, 20, 36, 20)
        main_layout.setSpacing(14)

        # Header
        header_layout = QHBoxLayout()
        header_layout.addStretch()

        # logo
        logo_lbl = QLabel()
        if os.path.exists(self.logo_path):
            logo_pix = QPixmap(self.logo_path).scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lbl.setPixmap(logo_pix)
        header_layout.addWidget(logo_lbl)
        header_layout.addSpacing(12)

        # title
        title_lbl = QLabel("Image Recommender")
        title_lbl.setFont(QFont("Arial", 40, QFont.Bold))
        title_lbl.setStyleSheet("color: #e60028;")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Input row: path + browse + search + image preview
        row = QHBoxLayout()
        row.addStretch()

        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Enter image path or click Browse …")
        self.input_path.setFixedWidth(420)
        row.addWidget(self.input_path)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.on_browse)
        row.addWidget(self.browse_btn)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.on_search)
        row.addWidget(self.search_btn)

        row.addSpacing(24)

        # Preview thumbnail
        self.preview = QLabel()
        self.preview.setFixedSize(300, 300)
        self.preview.setStyleSheet("border: 2px solid #ccc; background-color: white;")
        self.preview.setAlignment(Qt.AlignCenter)
        row.addWidget(self.preview)

        row.addStretch()
        main_layout.addLayout(row)

        # Results title + status
        self.results_title = QLabel("Top 5 Results:")
        self.results_title.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(self.results_title)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        main_layout.addWidget(self.status_label)

        # Grid of 5 result slots
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        for i in range(5):
            img_lbl = QLabel()
            img_lbl.setFixedSize(200, 200)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setStyleSheet("border: 1px solid #bbb; background-color: white;")

            score_lbl = QLabel("—")
            score_lbl.setAlignment(Qt.AlignCenter)
            score_lbl.setWordWrap(True)
            score_lbl.setFixedWidth(200)

            self.result_widgets.append((img_lbl, score_lbl))
            grid.addWidget(img_lbl, 0, i)
            grid.addWidget(score_lbl, 1, i)

        main_layout.addLayout(grid)
        main_layout.addStretch()

        # Important: setMinimumSize so painting scales smoothly on some window managers
        self.setMinimumSize(800, 600)


    def paintEvent(self, event):
        # Paint background scaled to window size so it always fills window.
        if not self.bg_pixmap.isNull():
            painter = QPainter(self)
            scaled = self.bg_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            # center the pixmap
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        super().paintEvent(event)


    def resizeEvent(self, event):
        # ensure position correctly and background repainted
        super().resizeEvent(event)
        self.update()


    # ------ UI actions ------

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.input_path.setText(path)
            self.display_preview(path)

    def display_preview(self, path):
        pix = QPixmap(path)
        if not pix.isNull():
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview.clear()

    def on_search(self):
        query = self.input_path.text().strip()
        if not query or not os.path.isfile(query):
            QMessageBox.warning(self, "No image", "Please select a valid image file to search.")
            return

        if not os.path.exists(self.CLIP_INDEX_PATH) or not os.path.exists(self.CLIP_MAPPING_PATH):
            QMessageBox.warning(self, "Missing index/mapping",
                                f"Could not find index or mapping:\n{self.CLIP_INDEX_PATH}\n{self.CLIP_MAPPING_PATH}")
            return

        # prepare UI
        self.search_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.status_label.setText("Searching — please wait...")
        QApplication.processEvents()

        # start background search
        self.search_thread = SearchThread(query, self.CLIP_INDEX_PATH, self.CLIP_MAPPING_PATH, k_clip=20, top_k_result=5)
        self.search_thread.finished.connect(self.on_search_finished)
        self.search_thread.error.connect(self.on_search_error)
        self.search_thread.start()

    def on_search_error(self, message):
        self.status_label.setText("")
        QMessageBox.critical(self, "Search error", message)
        self.search_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.search_thread = None

    def on_search_finished(self, results):
        # results is list[(path, score)]
        self.status_label.setText("")
        self.search_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)

        # clear old
        for img_lbl, score_lbl in self.result_widgets:
            img_lbl.clear()
            score_lbl.setText("—")

        for i, (path, score) in enumerate(results[:5]):
            img_lbl, score_lbl = self.result_widgets[i]
            if os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    img_lbl.setPixmap(pix.scaled(img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            score_lbl.setText(f"{os.path.basename(path)}\n{score:.4f}")

        self.search_thread = None


def main():
    # optional ghigh DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()
    QTimer.singleShot(0, win.showMaximized)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
