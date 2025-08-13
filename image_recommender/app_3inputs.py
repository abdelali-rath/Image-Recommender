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

    def __init__(self, query_paths, index_path, mapping_path, k_clip=20, top_k_result=5):
        super().__init__()
        # `query_paths` may be a string or a list of strings; we pass it directly
        self.query_paths = query_paths
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.k_clip = k_clip
        self.top_k_result = top_k_result

    def run(self):
        try:
            # Call the pipeline's combined_similarity_search.
            # Supports either a single path str or a list of paths.
            results = combined_similarity_search(
                self.query_paths,
                clip_index_path=self.index_path,
                clip_mapping_path=self.mapping_path,
                k_clip=self.k_clip,
                top_k_result=self.top_k_result
            )
            # Emit found results back to the main thread
            self.finished.emit(results)
        except Exception as e:
            # Send error message
            self.error.emit(str(e))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Assets and index, mapping paths
        self.logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logo.png"))
        self.background_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "App_background.jpg"))
        self.CLIP_INDEX_PATH = os.path.abspath("data/out/clip_index.ann")
        self.CLIP_MAPPING_PATH = os.path.abspath("data/out/index_to_id.json")

        # Load background pixmap once
        self.bg_pixmap = QPixmap(self.background_path) if os.path.exists(self.background_path) else QPixmap()

        self.search_thread = None
        # list of tuples: For up to 3 inputs
        self.input_widgets = []
        # list of (img_label, score_label) for result slots
        self.result_widgets = []

        self._setup_ui()

    def _setup_ui(self):
        # Window title and icon
        self.setWindowTitle("Image Recommender")
        if os.path.exists(self.logo_path):
            self.setWindowIcon(QIcon(self.logo_path))

        # Central widget + layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(36, 20, 36, 20)
        main_layout.setSpacing(14)

        # Header
        header_layout = QHBoxLayout()
        header_layout.addStretch()

        logo_lbl = QLabel()
        if os.path.exists(self.logo_path):
            logo_pix = QPixmap(self.logo_path).scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lbl.setPixmap(logo_pix)
        header_layout.addWidget(logo_lbl)
        header_layout.addSpacing(12)

        title_lbl = QLabel("Image Recommender")
        title_lbl.setFont(QFont("Arial", 40, QFont.Bold))
        title_lbl.setStyleSheet("color: #e60028;")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # INputs
        # 3 columns with line-edit, Browse button and thumbnail
        inputs_row = QHBoxLayout()
        inputs_row.setSpacing(20)
        inputs_row.addStretch()

        for i in range(3):
            col = QVBoxLayout()

            # Horizontal sub-row: path edit and browse
            path_row = QHBoxLayout()
            path_edit = QLineEdit()
            path_edit.setPlaceholderText(f"Image {i+1} (optional)")
            path_edit.setFixedWidth(260)
            browse_btn = QPushButton("Browse")
            # Capture the index "i" to route the callback to the correct slot
            browse_btn.clicked.connect(lambda checked, idx=i: self.on_browse(idx))

            path_row.addWidget(path_edit)
            path_row.addWidget(browse_btn)

            # Thumbnail preview for this input
            thumb = QLabel()
            thumb.setFixedSize(180, 180)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("border: 1px solid #ccc; background-color: white;")

            col.addLayout(path_row)
            col.addSpacing(8)
            col.addWidget(thumb, alignment=Qt.AlignHCenter)

            # Store widgets for later access
            self.input_widgets.append((path_edit, browse_btn, thumb))

            inputs_row.addLayout(col)

        inputs_row.addStretch()
        main_layout.addLayout(inputs_row)

        # Search controls (Search button placed under the inputs)
        controls_row = QHBoxLayout()
        controls_row.addStretch()
        self.search_btn = QPushButton("Search")
        self.search_btn.setFixedWidth(120)
        self.search_btn.clicked.connect(self.on_search)
        controls_row.addWidget(self.search_btn)
        controls_row.addStretch()
        main_layout.addLayout(controls_row)

        # Status / feedback label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Results header
        results_header = QLabel("Top 5 Results:")
        results_header.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(results_header)

        # Grid of 5 result slots, 1 row x 5 cols
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

        # Minimum size to avoid tiny windows
        self.setMinimumSize(900, 700)

    # Draw the background pixmap scaled and centered every paint
    def paintEvent(self, event):
        if not self.bg_pixmap.isNull():
            painter = QPainter(self)
            scaled = self.bg_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        super().paintEvent(event)

    def resizeEvent(self, event):
        # Trigger repaint so background scales nicely on resize
        super().resizeEvent(event)
        self.update()



    # INput handling

    def on_browse(self, idx: int):
    
        # Open file dialog for input slot "idx" (0..2) and update the corresponding QLineEdit and thumbnail preview.

        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        # Update UI controls for this index
        path_edit, _, thumb = self.input_widgets[idx]
        path_edit.setText(path)
        self.set_thumbnail(thumb, path)

    def set_thumbnail(self, thumb_label: QLabel, path: str):

        # Helper to set a scaled thumbnail on a QLabel; clears the label if image can't be read.
        pix = QPixmap(path)
        if not pix.isNull():
            thumb_label.setPixmap(pix.scaled(thumb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb_label.clear()

    # ------------- Search flow -------------

    def on_search(self):
        
        # Collect non empty input paths (1..3), validate them, start the SearchThread and block controls.
        # Collect non-empty, existing paths from the three inputs
        paths = []
        for path_edit, _, _ in self.input_widgets:
            p = path_edit.text().strip()
            if p:
                paths.append(p)

        if not paths:
            QMessageBox.warning(self, "No images", "Please provide at least one input image (1–3).")
            return

        # Validate files
        for p in paths:
            if not os.path.isfile(p):
                QMessageBox.warning(self, "Invalid file", f"File not found:\n{p}")
                return

        # Check index/mapping presence
        if not os.path.exists(self.CLIP_INDEX_PATH) or not os.path.exists(self.CLIP_MAPPING_PATH):
            QMessageBox.warning(
                self,
                "Missing index/mapping",
                f"Could not find index or mapping files:\n{self.CLIP_INDEX_PATH}\n{self.CLIP_MAPPING_PATH}"
            )
            return

        # Block controls and show status
        self.search_btn.setEnabled(False)
        for _, btn, _ in self.input_widgets:
            btn.setEnabled(False)
        for path_edit, _, _ in self.input_widgets:
            path_edit.setEnabled(False)

        self.status_label.setText("Searching — please wait...")
        QApplication.processEvents()  # let UI update immediately

        # start worker thread: combined_similarity_search accepts list[str]
        self.search_thread = SearchThread(
            query_paths=paths,
            index_path=self.CLIP_INDEX_PATH,
            mapping_path=self.CLIP_MAPPING_PATH,
            k_clip=20,
            top_k_result=5
        )
        self.search_thread.finished.connect(self.on_search_finished)
        self.search_thread.error.connect(self.on_search_error)
        self.search_thread.start()

    def on_search_error(self, message: str):
        # Re-enable UI and show error
        self.status_label.setText("")
        QMessageBox.critical(self, "Search error", message)
        self._restore_ui_after_search()

    def on_search_finished(self, results: list):
        
        # Called when the background search thread completes.
        # "results" is expected to be a list of (path, score).
        
        self.status_label.setText("")
        # clear previous results
        for img_lbl, score_lbl in self.result_widgets:
            img_lbl.clear()
            score_lbl.setText("—")

        # populate up to 5 results
        for i, (path, score) in enumerate(results[:5]):
            img_lbl, score_lbl = self.result_widgets[i]
            if os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    img_lbl.setPixmap(pix.scaled(img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # show filename + numeric score
            score_lbl.setText(f"{os.path.basename(path)}\n{score:.4f}")

        # restore UI controls
        self._restore_ui_after_search()
        self.search_thread = None

    def _restore_ui_after_search(self):
        # Re-enable buttons and edits after finish
        self.search_btn.setEnabled(True)
        for _, btn, _ in self.input_widgets:
            btn.setEnabled(True)
        for path_edit, _, _ in self.input_widgets:
            path_edit.setEnabled(True)




def main():
    # High DPI scaling for modern displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()
    QTimer.singleShot(0, win.showMaximized)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
