import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox,
    QSpinBox, QProgressBar, QAction, QMenu, QToolTip, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPainter, QDesktopServices
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QSettings, QSize

# Allow running as a script from repo
if __package__ is None and __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from image_recommender.pipeline.search_pipeline import combined_similarity_search

# ----------------------- Index/Mapping & Assets resolution -----------------------

def _resolve_index_and_mapping():
    here = Path(__file__).resolve()
    pkg_root = here.parent
    repo_root = pkg_root.parent

    env_idx = os.getenv("CLIP_INDEX_PATH")
    env_map = os.getenv("CLIP_MAPPING_PATH")
    if env_idx and env_map and Path(env_idx).exists() and Path(env_map).exists():
        return str(Path(env_idx)), str(Path(env_map))

    cand1 = pkg_root / "data" / "out"
    idx1, map1 = cand1 / "clip_index.ann", cand1 / "index_to_id.json"
    if idx1.exists() and map1.exists():
        return str(idx1), str(map1)

    cand2 = repo_root / "data" / "out"
    idx2, map2 = cand2 / "clip_index.ann", cand2 / "index_to_id.json"
    if idx2.exists() and map2.exists():
        return str(idx2), str(map2)

    # fallback to package path even if missing (UI will warn)
    return str(idx1), str(map1)

try:
    from importlib.resources import files as _ir_files  # Python 3.9+
except Exception:
    _ir_files = None

def _resolve_assets():
    here = Path(__file__).resolve()
    pkg_root = here.parent
    repo_root = pkg_root.parent

    env_logo = os.getenv("APP_LOGO_PATH")
    env_bg   = os.getenv("APP_BACKGROUND_PATH")

    def _pkg_res(mod, name):
        if _ir_files is None:
            return None
        try:
            return str(_ir_files(mod) / name)
        except Exception:
            return None

    def _first_existing(paths):
        for p in paths:
            if p and Path(p).exists():
                return str(Path(p))
        return ""

    logo_candidates = [
        env_logo,
        _pkg_res("image_recommender.assets", "logo.png"),
        _pkg_res("image_recommender.assets.images", "logo.png"),
        pkg_root / "assets" / "logo.png",
        pkg_root / "assets" / "images" / "logo.png",
        repo_root / "assets" / "logo.png",
        repo_root / "assets" / "images" / "logo.png",
        repo_root / "logo.png",
    ]
    bg_candidates = [
        env_bg,
        _pkg_res("image_recommender.assets", "app_background.jpg"),
        _pkg_res("image_recommender.assets.images", "app_background.jpg"),
        _pkg_res("image_recommender.assets", "App_background.jpg"),
        _pkg_res("image_recommender.assets.images", "App_background.jpg"),
        pkg_root / "assets" / "app_background.jpg",
        pkg_root / "assets" / "images" / "app_background.jpg",
        pkg_root / "assets" / "App_background.jpg",
        pkg_root / "assets" / "images" / "App_background.jpg",
        repo_root / "assets" / "app_background.jpg",
        repo_root / "assets" / "images" / "app_background.jpg",
        repo_root / "App_background.jpg",
        repo_root / "assets" / "images" / "App_background.jpg",
    ]

    return _first_existing(logo_candidates), _first_existing(bg_candidates)

# ----------------------- Small helpers -----------------------

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def is_image(path: str) -> bool:
    return str(path).lower().endswith(IMG_EXTS)

def human_score(score: float) -> str:
    # Adjust if your scores are similarity (higher=better) or distance (lower=better)
    return f"{score:.4f}"

# ----------------------- Custom widgets -----------------------

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

class DropLineEdit(QLineEdit):
    fileDropped = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        ok = False
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                if url.isLocalFile() and is_image(url.toLocalFile()):
                    ok = True
                    break
        if ok:
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                p = url.toLocalFile()
                if is_image(p):
                    self.setText(p)
                    self.fileDropped.emit(p)
                    break

class DropBox(QFrame):
    """Bordered drop area with optional click-to-open dialog; no hint text by default."""
    fileSelected = pyqtSignal(str)

    def __init__(self, size=QSize(220, 220), hint=None):
        super().__init__()
        self.setAcceptDrops(True)
        self._pix = None

        self.setFixedSize(size)
        self._normal_style = (
            "QFrame { border: 2px dashed #999; border-radius: 12px; "
            "background: rgba(255,255,255,0.9); }"
        )
        self._hover_style = (
            "QFrame { border: 2px dashed #e60028; border-radius: 12px; "
            "background: rgba(230,0,40,0.06); }"
        )
        self.setStyleSheet(self._normal_style)

        # Content
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(6)

        self._thumb = QLabel("", self)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setScaledContents(False)

        self._layout.addWidget(self._thumb, alignment=Qt.AlignCenter)

    # --- public api ---
    def reset(self):
        self._pix = None
        self._thumb.clear()

    def setImage(self, path: str):
        pix = QPixmap(path)
        if not pix.isNull():
            self._pix = pix
            inner = QSize(max(10, self.width() - 24), max(10, self.height() - 24))
            self._thumb.setPixmap(pix.scaled(inner, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.reset()

    # --- dnd / click ---
    def dragEnterEvent(self, e):
        ok = False
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.isLocalFile() and is_image(u.toLocalFile()):
                    ok = True
                    break
        if ok:
            self.setStyleSheet(self._hover_style)
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self._normal_style)
        super().dragLeaveEvent(e)

    def dropEvent(self, e):
        self.setStyleSheet(self._normal_style)
        for u in e.mimeData().urls():
            if u.isLocalFile():
                p = u.toLocalFile()
                if is_image(p):
                    self.setImage(p)
                    self.fileSelected.emit(p)
                    break
        e.acceptProposedAction()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select image", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
            )
            if path:
                self.setImage(path)
                self.fileSelected.emit(path)
        super().mouseReleaseEvent(e)

# ----------------------- Worker thread -----------------------

class SearchThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, query_paths, index_path, mapping_path, k_clip=20, top_k_result=5):
        super().__init__()
        self.query_paths = query_paths
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.k_clip = k_clip
        self.top_k_result = top_k_result

    def run(self):
        try:
            results = combined_similarity_search(
                self.query_paths,
                clip_index_path=self.index_path,
                clip_mapping_path=self.mapping_path,
                k_clip=self.k_clip,
                top_k_result=self.top_k_result
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# ----------------------- Main Window -----------------------

class MainWindow(QMainWindow):
    MAX_INPUTS = 3
    MAX_RESULTS = 5

    def __init__(self):
        super().__init__()

        # Settings
        self.settings = QSettings("AdeelLab", "ImageRecommenderApp")

        # Paths & assets
        self.logo_path, self.background_path = _resolve_assets()
        def_idx, def_map = _resolve_index_and_mapping()
        self.CLIP_INDEX_PATH = self.settings.value("index_path", def_idx, type=str)
        self.CLIP_MAPPING_PATH = self.settings.value("mapping_path", def_map, type=str)

        # UI state
        self.bg_enabled = True
        self.bg_pixmap = QPixmap(self.background_path) if os.path.exists(self.background_path) else QPixmap()
        self.input_widgets = []            # (line_edit, browse_btn, dropbox)
        self.result_widgets = []           # (img_label, score_label, path)
        self.search_thread = None
        self.discard_next_finish = False   # for cancel UX

        self._setup_ui()
        self._setup_menu()
        self._load_initial_warnings()

    # ----------- UI setup -----------

    def _setup_ui(self):
        self.setWindowTitle("Image Recommender")
        if os.path.exists(self.logo_path):
            self.setWindowIcon(QIcon(self.logo_path))
        self.setMinimumSize(1000, 720)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 16, 24, 16)
        main_layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        logo_lbl = QLabel()
        if os.path.exists(self.logo_path):
            logo_pix = QPixmap(self.logo_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lbl.setPixmap(logo_pix)
        header.addStretch()
        header.addWidget(logo_lbl)

        title = QLabel("Image Recommender")
        title.setFont(QFont("Arial", 36, QFont.Bold))
        title.setStyleSheet("color:#e60028;")
        header.addWidget(title)
        header.addStretch()
        main_layout.addLayout(header)

        # Inputs row
        inputs_row = QHBoxLayout()
        inputs_row.setSpacing(20)
        inputs_row.addStretch()

        for i in range(self.MAX_INPUTS):
            col = QVBoxLayout()
            path_row = QHBoxLayout()

            path_edit = DropLineEdit()
            path_edit.setPlaceholderText(f"Image {i+1} (optional) – drag & drop or Browse")
            path_edit.setFixedWidth(300)
            path_edit.fileDropped.connect(lambda p, idx=i: self._on_dropbox_selected(idx, p))

            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(lambda _, idx=i: self.on_browse(idx))

            clear_btn = QPushButton("✕")
            clear_btn.setFixedWidth(28)
            clear_btn.setToolTip("Clear this input")
            clear_btn.clicked.connect(lambda _, idx=i: self._clear_input(idx))

            path_row.addWidget(path_edit)
            path_row.addWidget(browse_btn)
            path_row.addWidget(clear_btn)

            drop = DropBox(size=QSize(220, 220))  # no hint text
            drop.fileSelected.connect(lambda p, idx=i: self._on_dropbox_selected(idx, p))

            col.addLayout(path_row)
            col.addSpacing(6)
            col.addWidget(drop, alignment=Qt.AlignHCenter)

            self.input_widgets.append((path_edit, browse_btn, drop))
            inputs_row.addLayout(col)

        inputs_row.addStretch()
        main_layout.addLayout(inputs_row)

        # Controls row
        ctr = QHBoxLayout()
        ctr.addStretch()

        label_k = QLabel("k_clip:")
        label_k.setStyleSheet("color: black;")
        ctr.addWidget(label_k)

        self.k_spin = QSpinBox()
        self.k_spin.setRange(5, 2000)
        self.k_spin.setValue(self.settings.value("k_clip", 20, type=int))
        self.k_spin.setToolTip("Candidates taken from CLIP ANN index before reranking")
        ctr.addWidget(self.k_spin)

        ctr.addSpacing(12)

        label_top = QLabel("Top-K results:")
        label_top.setStyleSheet("color: black;")
        ctr.addWidget(label_top)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, self.MAX_RESULTS)
        self.topk_spin.setValue(self.settings.value("top_k", 5, type=int))
        self.topk_spin.setToolTip("How many final results to display")
        ctr.addWidget(self.topk_spin)

        ctr.addSpacing(24)

        self.search_btn = QPushButton("Search")
        self.search_btn.setFixedWidth(140)
        self.search_btn.setStyleSheet("QPushButton { color: black; }")
        self.search_btn.clicked.connect(self.on_search)
        ctr.addWidget(self.search_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(120)
        self.cancel_btn.setStyleSheet("QPushButton { color: black; }")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel_search)
        ctr.addWidget(self.cancel_btn)

        ctr.addStretch()
        main_layout.addLayout(ctr)

        # Status + progress
        sp = QHBoxLayout()
        sp.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#555;")
        sp.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        sp.addWidget(self.progress)
        sp.addStretch()
        main_layout.addLayout(sp)

        # Results
        results_header = QLabel("Results")
        results_header.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(results_header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        for i in range(self.MAX_RESULTS):
            img_lbl = ClickableLabel()
            img_lbl.setFixedSize(220, 220)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setStyleSheet("border:1px solid #bbb; background:white;")
            img_lbl.clicked.connect(lambda idx=i: self._open_result(idx))

            score_lbl = QLabel("—")
            score_lbl.setAlignment(Qt.AlignCenter)
            score_lbl.setWordWrap(True)
            score_lbl.setFixedWidth(220)

            self.result_widgets.append([img_lbl, score_lbl, ""])
            grid.addWidget(img_lbl, 0, i)
            grid.addWidget(score_lbl, 1, i)
        main_layout.addLayout(grid)

        main_layout.addStretch()

    def _setup_menu(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        set_index = QAction("Set &Index…", self)
        set_index.triggered.connect(self._choose_index)
        file_menu.addAction(set_index)

        set_map = QAction("Set &Mapping…", self)
        set_map.triggered.connect(self._choose_mapping)
        file_menu.addAction(set_map)

        file_menu.addSeparator()
        clear_all = QAction("&Clear Inputs", self)
        clear_all.triggered.connect(self._clear_all_inputs)
        file_menu.addAction(clear_all)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View
        view_menu = menubar.addMenu("&View")
        toggle_bg = QAction("Toggle &Background", self, checkable=True)
        toggle_bg.setChecked(True)
        toggle_bg.triggered.connect(self._toggle_background)
        view_menu.addAction(toggle_bg)

        # Help
        help_menu = menubar.addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

        # Status bar hint
        self.statusBar().showMessage("Tip: Drag & drop images into the boxes.")

    def _load_initial_warnings(self):
        # Index/mapping check
        msgs = []
        if not os.path.exists(self.CLIP_INDEX_PATH):
            msgs.append(f"Index not found: {self.CLIP_INDEX_PATH}")
        if not os.path.exists(self.CLIP_MAPPING_PATH):
            msgs.append(f"Mapping not found: {self.CLIP_MAPPING_PATH}")
        if msgs:
            QMessageBox.warning(self, "Index/Mapping missing", "\n".join(msgs))

    # ----------- Background drawing -----------

    def paintEvent(self, event):
        if self.bg_enabled and not self.bg_pixmap.isNull():
            painter = QPainter(self)
            scaled = self.bg_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        super().paintEvent(event)

    # ----------- Inputs & dropbox -----------

    def _on_dropbox_selected(self, idx: int, path: str):
        edit, _, drop = self.input_widgets[idx]
        if path:
            edit.setText(path)
            self.settings.setValue("last_dir", str(Path(path).parent))

    def on_browse(self, idx: int):
        last_dir = self.settings.value("last_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", last_dir, "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        edit, _, drop = self.input_widgets[idx]
        edit.setText(path)
        drop.setImage(path)

    def _clear_input(self, idx: int):
        edit, _, drop = self.input_widgets[idx]
        edit.clear()
        drop.reset()

    def _clear_all_inputs(self):
        for i in range(self.MAX_INPUTS):
            self._clear_input(i)
        for lbl, score, _ in self.result_widgets:
            lbl.clear()
            score.setText("—")

    # ----------- Search flow -----------

    def on_search(self):
        # Collect valid image paths
        paths = []
        for edit, _, _ in self.input_widgets:
            p = edit.text().strip()
            if p:
                if not os.path.isfile(p):
                    QMessageBox.warning(self, "Invalid file", f"File not found:\n{p}")
                    return
                if not is_image(p):
                    QMessageBox.warning(self, "Invalid image", f"Not an image file:\n{p}")
                    return
                paths.append(p)

        if not paths:
            QMessageBox.warning(self, "No images", "Please provide at least one input image (1–3).")
            return

        if not os.path.exists(self.CLIP_INDEX_PATH) or not os.path.exists(self.CLIP_MAPPING_PATH):
            QMessageBox.warning(
                self,
                "Missing index/mapping",
                f"Could not find index or mapping files:\n{self.CLIP_INDEX_PATH}\n{self.CLIP_MAPPING_PATH}"
            )
            return

        k_clip = int(self.k_spin.value())
        top_k = int(self.topk_spin.value())
        self.settings.setValue("k_clip", k_clip)
        self.settings.setValue("top_k", top_k)

        # Block controls + busy progress
        self._set_controls_enabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # busy
        self.status_label.setText("Searching…")

        # Start thread
        self.discard_next_finish = False
        self.search_thread = SearchThread(
            query_paths=paths,
            index_path=self.CLIP_INDEX_PATH,
            mapping_path=self.CLIP_MAPPING_PATH,
            k_clip=k_clip,
            top_k_result=top_k
        )
        self.search_thread.finished.connect(self.on_search_finished)
        self.search_thread.error.connect(self.on_search_error)
        self.search_thread.start()

    def on_cancel_search(self):
        # We can't force-kill the thread safely; instead, ignore its result and re-enable UI.
        if self.search_thread and self.search_thread.isRunning():
            self.discard_next_finish = True
            self.status_label.setText("Cancelled.")
            self._finish_search_ui()

    def on_search_error(self, message: str):
        if not self.discard_next_finish:
            self.status_label.setText("")
            QMessageBox.critical(self, "Search error", message)
        self._finish_search_ui()

    def on_search_finished(self, results: list):
        # Ignore stale results if user pressed Cancel
        if self.discard_next_finish:
            return

        self.status_label.setText("")
        # Clear previous
        for img_lbl, score_lbl, _ in self.result_widgets:
            img_lbl.clear()
            score_lbl.setText("—")

        # Populate
        for i, item in enumerate(results[: self.MAX_RESULTS]):
            try:
                path, score = item
            except Exception:
                if isinstance(item, dict) and "path" in item:
                    path = item["path"]
                    score = item.get("score", 0.0)
                else:
                    continue
            img_lbl, score_lbl, _ = self.result_widgets[i]
            self.result_widgets[i][2] = path  # store path for click
            if os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    img_lbl.setPixmap(pix.scaled(img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            base = os.path.basename(path)
            score_lbl.setText(f"{base}\n{human_score(score)}")
            score_lbl.setToolTip(path)

        self._finish_search_ui()

    def _finish_search_ui(self):
        self._set_controls_enabled(True)
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)  # reset
        self.search_thread = None
        self.discard_next_finish = False

    def _set_controls_enabled(self, enabled: bool):
        self.search_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        for edit, btn, _ in self.input_widgets:
            edit.setEnabled(enabled)
            btn.setEnabled(enabled)
        self.k_spin.setEnabled(enabled)
        self.topk_spin.setEnabled(enabled)

    # ----------- Result actions -----------

    def _open_result(self, idx: int):
        path = self.result_widgets[idx][2]
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QToolTip.showText(self.mapToGlobal(self.rect().center()), "No result here yet")

    # ----------- Menu actions -----------

    def _choose_index(self):
        start_dir = str(Path(self.CLIP_INDEX_PATH).parent) if self.CLIP_INDEX_PATH else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select CLIP index (.ann)", start_dir, "ANN files (*.ann);;All files (*)")
        if path:
            self.CLIP_INDEX_PATH = path
            self.settings.setValue("index_path", path)
            self.statusBar().showMessage(f"Index set: {path}", 4000)

    def _choose_mapping(self):
        start_dir = str(Path(self.CLIP_MAPPING_PATH).parent) if self.CLIP_MAPPING_PATH else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select mapping (index_to_id.json)", start_dir, "JSON files (*.json);;All files (*)")
        if path:
            self.CLIP_MAPPING_PATH = path
            self.settings.setValue("mapping_path", path)
            self.statusBar().showMessage(f"Mapping set: {path}", 4000)

    def _toggle_background(self, checked: bool):
        self.bg_enabled = checked
        self.update()

    def _about(self):
        QMessageBox.information(
            self,
            "About",
            "Image Recommender UI\n\n"
            "• Drag & drop boxes\n"
            "• Adjustable k_clip & Top-K\n"
            "• Remembers your settings\n\n"
            "© 2025"
        )

# ----------------------- App entry -----------------------

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    QTimer.singleShot(0, win.showMaximized)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
