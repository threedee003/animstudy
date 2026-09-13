
from __future__ import annotations

import csv
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import av
except ImportError:  # pragma: no cover
    av = None

from PyQt6.QtCore import (
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# --------------------------------------------------------------------------
# Palette: a neutral graphite surround so nothing competes with the artwork,
# with amber for the interface accent and the red/cyan pair that animators
# already read as "previous drawing" / "next drawing".
# --------------------------------------------------------------------------
BG_DEEP = "#1a1b1d"
BG_PANEL = "#232528"
BG_RAISED = "#2c2f33"
FG_MAIN = "#e6e3dd"
FG_MUTED = "#8e918f"
ACCENT = "#e0a33e"
NEW_DRAWING = "#e0a33e"
HELD_FRAME = "#4a4d51"
ONION_PREV = (1.00, 0.30, 0.28)
ONION_NEXT = (0.26, 0.72, 1.00)

STYLESHEET = f"""
QWidget {{
    background-color: {BG_PANEL};
    color: {FG_MAIN};
    font-size: 12px;
}}
QMainWindow, QSplitter {{ background-color: {BG_DEEP}; }}
QGroupBox {{
    border: 1px solid #3a3d42;
    border-radius: 3px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {FG_MUTED};
}}
QPushButton, QToolButton {{
    background-color: {BG_RAISED};
    border: 1px solid #3d4046;
    border-radius: 3px;
    padding: 5px 10px;
}}
QPushButton:hover, QToolButton:hover {{ background-color: #35383d; }}
QPushButton:pressed, QToolButton:pressed {{ background-color: #1f2124; }}
QPushButton:disabled {{ color: #5c5f63; border-color: #303338; }}
QPushButton#primary {{
    background-color: {ACCENT};
    color: #201703;
    border: none;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: #eeb75a; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_DEEP};
    border: 1px solid #3d4046;
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
    selection-color: #201703;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QListWidget {{
    background-color: {BG_DEEP};
    border: 1px solid #33363b;
}}
QListWidget::item {{ color: {FG_MUTED}; }}
QListWidget::item:selected {{
    background-color: #3a3225;
    border: 1px solid {ACCENT};
    color: {FG_MAIN};
}}
QProgressBar {{
    background-color: {BG_DEEP};
    border: 1px solid #3d4046;
    border-radius: 3px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; }}
QLabel#hint {{ color: {FG_MUTED}; }}
QLabel#stat {{ font-family: monospace; }}
QStatusBar {{ background-color: {BG_DEEP}; color: {FG_MUTED}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {BG_DEEP}; border: none; }}
QScrollBar::handle {{ background: #43474d; border-radius: 4px; }}
QScrollBar::handle:hover {{ background: #565b62; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
"""


# --------------------------------------------------------------------------
# Timecode helpers
# --------------------------------------------------------------------------
TIMECODE_RE = re.compile(
    r"^\s*(?:(?P<h>\d+):)?(?:(?P<m>\d+):)?(?P<s>\d+(?:\.\d+)?)\s*$"
)
FRAME_RE = re.compile(r"^\s*[fF]\s*(\d+)\s*$")


def parse_time(text: str, fps: float) -> Optional[float]:
    """Accepts 12.5, 1:05, 00:01:05.250, or f1234 (frame number)."""
    if not text:
        return None
    m = FRAME_RE.match(text)
    if m and fps > 0:
        return int(m.group(1)) / fps
    m = TIMECODE_RE.match(text)
    if not m:
        return None
    h = int(m.group("h") or 0)
    mi = int(m.group("m") or 0)
    if m.group("m") is None and m.group("h") is not None:
        h, mi = 0, h
    return h * 3600 + mi * 60 + float(m.group("s"))


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# --------------------------------------------------------------------------
# Frame model
# --------------------------------------------------------------------------
@dataclass
class Frame:
    local_index: int          # position inside the loaded range
    source_index: int         # frame number in the original file
    time_s: float             # presentation time in the original file
    rgb: np.ndarray           # HxWx3 uint8, contiguous
    signature: np.ndarray = field(repr=False, default=None)  # small grayscale
    is_new_drawing: bool = True
    drawing_number: int = 1
    hold_length: int = 1
    changed_fraction: float = 0.0
    _thumb: Optional[QPixmap] = field(default=None, repr=False)

    def thumbnail(self, height: int = 76) -> QPixmap:
        if self._thumb is None or self._thumb.height() != height:
            img = to_qimage(self.rgb)
            self._thumb = QPixmap.fromImage(
                img.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)
            )
        return self._thumb


def to_qimage(rgb: np.ndarray) -> QImage:
    """Copy into a QImage so the numpy buffer lifetime never matters."""
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def make_signature(rgb: np.ndarray, target_width: int = 192) -> np.ndarray:
    """Small grayscale fingerprint used for held-frame detection."""
    h, w, _ = rgb.shape
    step = max(1, int(round(w / target_width)))
    small = rgb[::step, ::step, :].astype(np.float32)
    return small[..., 0] * 0.299 + small[..., 1] * 0.587 + small[..., 2] * 0.114


# --------------------------------------------------------------------------
# Media probing and decoding
# --------------------------------------------------------------------------
@dataclass
class MediaInfo:
    path: str
    fps: float
    duration_s: float
    width: int
    height: int
    codec: str
    frame_count: int


def probe(path: str) -> MediaInfo:
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = 0.0
        count = stream.frames or (int(round(duration * fps)) if duration else 0)
        return MediaInfo(
            path=path,
            fps=fps,
            duration_s=duration,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            codec=stream.codec_context.name,
            frame_count=count,
        )


class DecodeWorker(QThread):
    """Decodes one time range into memory. Owns its own container."""

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, path, start_s, end_s, long_edge, max_frames, parent=None):
        super().__init__(parent)
        self.path = path
        self.start_s = start_s
        self.end_s = end_s
        self.long_edge = long_edge          # 0 = keep native resolution
        self.max_frames = max_frames
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            frames: List[Frame] = []
            with av.open(self.path) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                tb = stream.time_base
                fps = float(stream.average_rate) if stream.average_rate else 24.0
                expected = max(1, int(round((self.end_s - self.start_s) * fps)))

                seek_to = max(0.0, self.start_s - 1.0)
                container.seek(int(seek_to / tb), stream=stream, backward=True)

                for packet_frame in container.decode(stream):
                    if self._cancel:
                        self.failed.emit("Load cancelled.")
                        return
                    if packet_frame.pts is None:
                        continue
                    t = float(packet_frame.pts * tb)
                    if t < self.start_s - 1e-6:
                        continue
                    if t > self.end_s + 1e-6:
                        break

                    w, h = packet_frame.width, packet_frame.height
                    if self.long_edge and max(w, h) > self.long_edge:
                        scale = self.long_edge / max(w, h)
                        nw = max(2, int(round(w * scale)))
                        nh = max(2, int(round(h * scale)))
                        conv = packet_frame.reformat(
                            width=nw, height=nh, format="rgb24"
                        )
                    else:
                        conv = packet_frame.reformat(format="rgb24")
                    rgb = np.ascontiguousarray(conv.to_ndarray())

                    frames.append(
                        Frame(
                            local_index=len(frames),
                            source_index=int(round(t * fps)),
                            time_s=t,
                            rgb=rgb,
                            signature=make_signature(rgb),
                        )
                    )
                    self.progress.emit(len(frames), expected)
                    if len(frames) >= self.max_frames:
                        break

            if not frames:
                self.failed.emit(
                    "No frames decoded in that range. Check the start and end times."
                )
                return
            self.finished_ok.emit(frames)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Timing analysis
# --------------------------------------------------------------------------
def analyse_exposure(frames: List[Frame], threshold_pct: float, level: int = 8):
    """Mark which frames carry a new drawing and how long each is held."""
    if not frames:
        return
    frames[0].is_new_drawing = True
    frames[0].changed_fraction = 1.0

    for i in range(1, len(frames)):
        a = frames[i - 1].signature
        b = frames[i].signature
        if a.shape != b.shape:
            frac = 1.0
        else:
            frac = float(np.mean(np.abs(a - b) > level))
        frames[i].changed_fraction = frac
        frames[i].is_new_drawing = frac * 100.0 >= threshold_pct

    drawing = 0
    run_start = 0
    for i, f in enumerate(frames):
        if f.is_new_drawing:
            drawing += 1
            if i > 0:
                for j in range(run_start, i):
                    frames[j].hold_length = i - run_start
            run_start = i
        f.drawing_number = drawing
    for j in range(run_start, len(frames)):
        frames[j].hold_length = len(frames) - run_start


def exposure_summary(frames: List[Frame]) -> str:
    if not frames:
        return "No frames loaded."
    runs = []
    current = 0
    for f in frames:
        if f.is_new_drawing:
            if current:
                runs.append(current)
            current = 1
        else:
            current += 1
    if current:
        runs.append(current)

    total_frames = len(frames)
    total_drawings = len(runs)
    counts = {}
    for r in runs:
        counts[r] = counts.get(r, 0) + 1

    lines = [
        f"{total_frames} frames, {total_drawings} distinct drawings",
        f"average exposure {total_frames / max(1, total_drawings):.2f} frames per drawing",
        "",
        "held for   drawings   share of frames",
    ]
    names = {1: "ones", 2: "twos", 3: "threes", 4: "fours"}
    for length in sorted(counts):
        n = counts[length]
        share = 100.0 * n * length / total_frames
        tag = names.get(length, f"{length}s")
        lines.append(f"{length:>2} fr ({tag:<6}) {n:>6}      {share:5.1f}%")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Compositing: normal, onion skin, difference
# --------------------------------------------------------------------------
def compose(frames: List[Frame], idx: int, mode: str, onion_depth: int,
            onion_strength: float, diff_gain: float) -> np.ndarray:
    base = frames[idx].rgb
    if mode == "Normal" or len(frames) == 1:
        return base

    if mode == "Difference":
        if idx == 0:
            return base
        prev = frames[idx - 1].rgb.astype(np.int16)
        d = np.abs(base.astype(np.int16) - prev).sum(axis=2).astype(np.float32)
        d = np.clip(d * diff_gain, 0, 255)
        out = (base.astype(np.float32) * 0.18)
        out[..., 0] = np.clip(out[..., 0] + d, 0, 255)
        out[..., 1] = np.clip(out[..., 1] + d * 0.25, 0, 255)
        return out.astype(np.uint8)

    # Onion skin: neighbours are tinted, faded toward white, then darken-blended
    # so a white paper background stays white and only the ghost lines show.
    acc = base.astype(np.float32)
    for k in range(1, onion_depth + 1):
        alpha = onion_strength / k
        for offset, tint in ((-k, ONION_PREV), (k, ONION_NEXT)):
            j = idx + offset
            if j < 0 or j >= len(frames):
                continue
            nb = frames[j].rgb
            if nb.shape != base.shape:
                continue
            g = (nb[..., 0] * 0.299 + nb[..., 1] * 0.587 + nb[..., 2] * 0.114)
            tinted = np.stack([g * tint[0], g * tint[1], g * tint[2]], axis=-1)
            faded = 255.0 - (255.0 - tinted) * alpha
            acc = np.minimum(acc, faded)
    return np.clip(acc, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Canvas with wheel zoom and drag pan
# --------------------------------------------------------------------------
class FrameCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setBackgroundBrush(QColor("#141517"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._fit_on_next = True
        self._zoom = 1.0

    def set_image(self, qimg: QImage):
        changed = (
            self._item.pixmap().isNull()
            or self._item.pixmap().size() != qimg.size()
        )
        self._item.setPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(QRectF(self._item.pixmap().rect()))
        if changed or self._fit_on_next:
            self.fit()

    def fit(self):
        if self._item.pixmap().isNull():
            return
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_on_next = False
        self._zoom = 1.0

    def zoom_to(self, factor: float):
        if self._item.pixmap().isNull():
            return
        self.resetTransform()
        self.scale(factor, factor)
        self._zoom = factor
        self._fit_on_next = False

    def wheelEvent(self, event):
        if self._item.pixmap().isNull():
            return
        step = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(step, step)
        self._zoom *= step
        self._fit_on_next = False


# --------------------------------------------------------------------------
# Exposure strip: one tick per frame, amber for a new drawing
# --------------------------------------------------------------------------
class ExposureStrip(QWidget):
    seek_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames: List[Frame] = []
        self.current = 0
        self.setMinimumHeight(34)
        self.setMaximumHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Exposure sheet. Tall amber ticks are new drawings.")

    def set_frames(self, frames: List[Frame]):
        self.frames = frames
        self.update()

    def set_current(self, idx: int):
        self.current = idx
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG_DEEP))
        n = len(self.frames)
        if n == 0:
            p.setPen(QColor(FG_MUTED))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load a range to see the exposure sheet",
            )
            return

        w = self.width()
        step = w / n
        for i, f in enumerate(self.frames):
            x = i * step
            bw = max(1.0, step - (0.6 if step > 3 else 0.0))
            if f.is_new_drawing:
                p.fillRect(
                    QRectF(x, 4, bw, self.height() - 8), QColor(NEW_DRAWING)
                )
            else:
                p.fillRect(
                    QRectF(x, self.height() * 0.55, bw, self.height() * 0.28),
                    QColor(HELD_FRAME),
                )
        p.setPen(QPen(QColor("#ffffff"), 2))
        cx = self.current * step + step / 2
        p.drawLine(int(cx), 0, int(cx), self.height())
        p.end()

    def mousePressEvent(self, event):
        if not self.frames:
            return
        idx = int(event.position().x() / max(1e-6, self.width() / len(self.frames)))
        self.seek_requested.emit(max(0, min(len(self.frames) - 1, idx)))

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.mousePressEvent(event)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class AnimStudy(QMainWindow):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("AnimStudy")
        self.resize(1500, 940)

        self.info: Optional[MediaInfo] = None
        self.frames: List[Frame] = []
        self.current = 0
        self.worker: Optional[DecodeWorker] = None

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._build_shortcuts()
        self._set_loaded_state(False)

        if initial_path:
            self.open_path(initial_path)

    # ---------------- UI construction ----------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_source_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_viewer_side())
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1120, 340])
        outer.addWidget(splitter, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("Open a movie to begin.")

    def _build_source_bar(self) -> QWidget:
        box = QGroupBox("Source and range")
        lay = QHBoxLayout(box)
        lay.setSpacing(10)

        self.open_btn = QPushButton("Open movie")
        self.open_btn.clicked.connect(self.choose_file)
        lay.addWidget(self.open_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setObjectName("hint")
        self.file_label.setMinimumWidth(180)
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        lay.addWidget(self.file_label, 1)

        self.start_edit = QLineEdit("00:00:00.000")
        self.end_edit = QLineEdit("00:00:04.000")
        for e in (self.start_edit, self.end_edit):
            e.setFixedWidth(108)
            e.setToolTip("12.5  |  1:05  |  00:01:05.250  |  f1234 for a frame number")

        lay.addWidget(QLabel("From"))
        lay.addWidget(self.start_edit)
        lay.addWidget(QLabel("to"))
        lay.addWidget(self.end_edit)

        self.long_edge_spin = QSpinBox()
        self.long_edge_spin.setRange(0, 4096)
        self.long_edge_spin.setSingleStep(64)
        self.long_edge_spin.setValue(1280)
        self.long_edge_spin.setSpecialValueText("native")
        self.long_edge_spin.setFixedWidth(86)
        self.long_edge_spin.setToolTip(
            "Longest edge held in memory. 0 keeps the source resolution."
        )
        lay.addWidget(QLabel("Detail"))
        lay.addWidget(self.long_edge_spin)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(2, 5000)
        self.max_frames_spin.setValue(600)
        self.max_frames_spin.setFixedWidth(76)
        self.max_frames_spin.setToolTip("Safety cap on how many frames are decoded.")
        lay.addWidget(QLabel("Cap"))
        lay.addWidget(self.max_frames_spin)

        self.load_btn = QPushButton("Load range")
        self.load_btn.setObjectName("primary")
        self.load_btn.clicked.connect(self.load_range)
        lay.addWidget(self.load_btn)
        return box

    def _build_viewer_side(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.pages = QStackedWidget()
        self.canvas = FrameCanvas()
        self.pages.addWidget(self.canvas)

        self.sheet = QListWidget()
        self.sheet.setViewMode(QListView.ViewMode.IconMode)
        self.sheet.setResizeMode(QListView.ResizeMode.Adjust)
        self.sheet.setMovement(QListView.Movement.Static)
        self.sheet.setSpacing(6)
        self.sheet.setWordWrap(True)
        self.sheet.setIconSize(QSize(210, 122))
        self.sheet.setGridSize(QSize(226, 168))
        self.sheet.currentRowChanged.connect(self._sheet_selected)
        self.pages.addWidget(self.sheet)
        lay.addWidget(self.pages, 1)

        self.strip_bar = ExposureStrip()
        self.strip_bar.seek_requested.connect(self.goto)
        lay.addWidget(self.strip_bar)

        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListView.ViewMode.IconMode)
        self.filmstrip.setFlow(QListView.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setMovement(QListView.Movement.Static)
        self.filmstrip.setIconSize(QSize(128, 76))
        self.filmstrip.setGridSize(QSize(136, 108))
        self.filmstrip.setFixedHeight(136)
        self.filmstrip.setHorizontalScrollMode(
            QListWidget.ScrollMode.ScrollPerPixel
        )
        self.filmstrip.currentRowChanged.connect(self._strip_selected)
        lay.addWidget(self.filmstrip)

        lay.addWidget(self._build_transport())
        return panel

    def _build_transport(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        def tool(text, slot, tip):
            b = QToolButton()
            b.setText(text)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            return b

        self.btn_first = tool("|<", lambda: self.goto(0), "First frame (Home)")
        self.btn_prev_draw = tool(
            "<<", self.prev_drawing, "Previous new drawing (P)"
        )
        self.btn_prev = tool("<", lambda: self.step(-1), "Back one frame (Left)")
        self.btn_play = tool("Play", self.toggle_play, "Play or pause (Space)")
        self.btn_next = tool(">", lambda: self.step(1), "Forward one frame (Right)")
        self.btn_next_draw = tool(">>", self.next_drawing, "Next new drawing (N)")
        self.btn_last = tool(
            ">|", lambda: self.goto(len(self.frames) - 1), "Last frame (End)"
        )
        for b in (
            self.btn_first,
            self.btn_prev_draw,
            self.btn_prev,
            self.btn_play,
            self.btn_next,
            self.btn_next_draw,
            self.btn_last,
        ):
            lay.addWidget(b)

        lay.addSpacing(12)
        lay.addWidget(QLabel("Speed"))
        self.speed_combo = QComboBox()
        for label in ("1x", "1/2x", "1/4x", "1/8x", "2x"):
            self.speed_combo.addItem(label)
        self.speed_combo.currentIndexChanged.connect(self._retime_playback)
        lay.addWidget(self.speed_combo)

        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)
        lay.addWidget(self.loop_check)

        lay.addStretch(1)

        self.view_toggle = QPushButton("Contact sheet")
        self.view_toggle.setCheckable(True)
        self.view_toggle.clicked.connect(self._toggle_view)
        self.view_toggle.setToolTip("Switch between the single frame and the grid (G)")
        lay.addWidget(self.view_toggle)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self.canvas.fit)
        lay.addWidget(self.fit_btn)
        self.hundred_btn = QPushButton("100%")
        self.hundred_btn.clicked.connect(lambda: self.canvas.zoom_to(1.0))
        lay.addWidget(self.hundred_btn)
        return bar

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(10)

        # Media facts
        media_box = QGroupBox("Movie")
        form = QFormLayout(media_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_res = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_dur = QLabel("-")
        self.lbl_codec = QLabel("-")
        for label, widget in (
            ("Resolution", self.lbl_res),
            ("Frame rate", self.lbl_fps),
            ("Duration", self.lbl_dur),
            ("Codec", self.lbl_codec),
        ):
            widget.setObjectName("stat")
            form.addRow(label, widget)
        lay.addWidget(media_box)

        # Current frame
        frame_box = QGroupBox("Current frame")
        fform = QFormLayout(frame_box)
        fform.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_pos = QLabel("-")
        self.lbl_src = QLabel("-")
        self.lbl_tc = QLabel("-")
        self.lbl_drawing = QLabel("-")
        self.lbl_hold = QLabel("-")
        self.lbl_change = QLabel("-")
        for label, widget in (
            ("In range", self.lbl_pos),
            ("Source frame", self.lbl_src),
            ("Timecode", self.lbl_tc),
            ("Drawing", self.lbl_drawing),
            ("Held for", self.lbl_hold),
            ("Pixels changed", self.lbl_change),
        ):
            widget.setObjectName("stat")
            fform.addRow(label, widget)
        lay.addWidget(frame_box)

        # Display mode
        disp_box = QGroupBox("View")
        dform = QFormLayout(disp_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Normal", "Onion skin", "Difference"])
        self.mode_combo.currentIndexChanged.connect(self.refresh_frame)
        dform.addRow("Mode", self.mode_combo)

        self.onion_depth = QSpinBox()
        self.onion_depth.setRange(1, 6)
        self.onion_depth.setValue(2)
        self.onion_depth.valueChanged.connect(self.refresh_frame)
        dform.addRow("Onion depth", self.onion_depth)

        self.onion_strength = QDoubleSpinBox()
        self.onion_strength.setRange(0.05, 1.0)
        self.onion_strength.setSingleStep(0.05)
        self.onion_strength.setValue(0.45)
        self.onion_strength.valueChanged.connect(self.refresh_frame)
        dform.addRow("Onion strength", self.onion_strength)

        self.diff_gain = QDoubleSpinBox()
        self.diff_gain.setRange(0.2, 20.0)
        self.diff_gain.setSingleStep(0.5)
        self.diff_gain.setValue(3.0)
        self.diff_gain.valueChanged.connect(self.refresh_frame)
        dform.addRow("Difference gain", self.diff_gain)

        hint = QLabel(
            "Onion skin tints the previous drawings red and the next ones blue. "
            "Difference highlights only what moved since the frame before."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        dform.addRow(hint)
        lay.addWidget(disp_box)

        # Timing
        timing_box = QGroupBox("Timing")
        tlay = QVBoxLayout(timing_box)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Sensitivity"))
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.001, 20.0)
        self.sensitivity.setDecimals(3)
        self.sensitivity.setSingleStep(0.05)
        self.sensitivity.setValue(0.200)
        self.sensitivity.setSuffix(" %")
        self.sensitivity.setToolTip(
            "A frame counts as a new drawing when at least this share of pixels "
            "changed. Raise it if compression noise is splitting held frames."
        )
        trow.addWidget(self.sensitivity)
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self.recompute_timing)
        trow.addWidget(recompute)
        tlay.addLayout(trow)

        self.timing_stats = QLabel("No frames loaded.")
        self.timing_stats.setObjectName("stat")
        self.timing_stats.setFont(QFont("monospace", 10))
        self.timing_stats.setWordWrap(False)
        self.timing_stats.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        tlay.addWidget(self.timing_stats)
        lay.addWidget(timing_box)

        # Export
        exp_box = QGroupBox("Export")
        elay = QVBoxLayout(exp_box)
        b1 = QPushButton("Frames as PNG sequence")
        b1.clicked.connect(self.export_sequence)
        b2 = QPushButton("Contact sheet as PNG")
        b2.clicked.connect(self.export_contact_sheet)
        b3 = QPushButton("Exposure sheet as CSV")
        b3.clicked.connect(self.export_csv)
        for b in (b1, b2, b3):
            elay.addWidget(b)
        note = QLabel(
            "Exports use the loaded detail setting. Set detail to native "
            "before loading if you need full resolution stills."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        elay.addWidget(note)
        lay.addWidget(exp_box)

        keys = QLabel(
            "Left / Right  step one frame\n"
            "Shift + arrow  step ten\n"
            "N / P  next or previous drawing\n"
            "Space  play or pause\n"
            "O  onion skin    D  difference\n"
            "G  contact sheet    F  fit\n"
            "Home / End  range ends"
        )
        keys.setObjectName("hint")
        keys.setFont(QFont("monospace", 10))
        lay.addWidget(keys)

        lay.addStretch(1)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(320)
        return scroll

    def _build_shortcuts(self):
        def sc(seq, slot):
            s = QShortcut(QKeySequence(seq), self)
            s.activated.connect(slot)
            return s

        sc(Qt.Key.Key_Right, lambda: self.step(1))
        sc(Qt.Key.Key_Left, lambda: self.step(-1))
        sc("Shift+Right", lambda: self.step(10))
        sc("Shift+Left", lambda: self.step(-10))
        sc(Qt.Key.Key_Period, lambda: self.step(1))
        sc(Qt.Key.Key_Comma, lambda: self.step(-1))
        sc(Qt.Key.Key_Home, lambda: self.goto(0))
        sc(Qt.Key.Key_End, lambda: self.goto(len(self.frames) - 1))
        sc(Qt.Key.Key_Space, self.toggle_play)
        sc("N", self.next_drawing)
        sc("P", self.prev_drawing)
        sc("O", lambda: self._cycle_mode("Onion skin"))
        sc("D", lambda: self._cycle_mode("Difference"))
        sc("F", self.canvas.fit)
        sc("G", self._shortcut_toggle_view)
        sc("Ctrl+O", self.choose_file)

    # ---------------- file handling ----------------
    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open a movie",
            os.path.expanduser("~"),
            "Video (*.mkv *.mp4 *.mov *.avi *.webm *.ogv *.m4v *.mpg *.mpeg);;All files (*)",
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        if av is None:
            QMessageBox.critical(
                self, "PyAV missing", "Install the decoder first:\n\npip install av"
            )
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Not found", f"No file at {path}")
            return
        try:
            self.info = probe(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot read that file", str(exc))
            return

        self.file_label.setText(os.path.basename(path))
        self.file_label.setToolTip(path)
        self.lbl_res.setText(f"{self.info.width} x {self.info.height}")
        self.lbl_fps.setText(f"{self.info.fps:.3f} fps")
        self.lbl_dur.setText(
            f"{format_time(self.info.duration_s)}  ({self.info.frame_count} frames)"
        )
        self.lbl_codec.setText(self.info.codec)
        self.start_edit.setText("00:00:00.000")
        default_end = min(self.info.duration_s, 4.0) if self.info.duration_s else 4.0
        self.end_edit.setText(format_time(default_end))
        self.status.showMessage(
            f"Loaded {os.path.basename(path)}. Choose a range and press Load range."
        )

    def load_range(self):
        if not self.info:
            QMessageBox.information(self, "No movie", "Open a movie first.")
            return
        start = parse_time(self.start_edit.text(), self.info.fps)
        end = parse_time(self.end_edit.text(), self.info.fps)
        if start is None or end is None:
            QMessageBox.warning(
                self,
                "Times not understood",
                "Use 12.5, 1:05, 00:01:05.250, or f1234 for a frame number.",
            )
            return
        if end <= start:
            QMessageBox.warning(
                self, "Empty range", "The end time must come after the start time."
            )
            return

        self.stop_playback()
        self.load_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.showMessage("Decoding...")

        self.worker = DecodeWorker(
            self.info.path,
            start,
            end,
            self.long_edge_spin.value(),
            self.max_frames_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_frames_ready)
        self.worker.failed.connect(self._on_decode_failed)
        self.worker.start()

    def _on_progress(self, done: int, expected: int):
        self.progress.setMaximum(max(done, expected))
        self.progress.setValue(done)

    def _on_decode_failed(self, message: str):
        self.progress.setVisible(False)
        self.load_btn.setEnabled(True)
        self.status.showMessage(message)
        QMessageBox.warning(self, "Load failed", message)

    def _on_frames_ready(self, frames: List[Frame]):
        self.progress.setVisible(False)
        self.load_btn.setEnabled(True)
        self.frames = frames
        self.current = 0
        analyse_exposure(self.frames, self.sensitivity.value())
        self._populate_strips()
        self.strip_bar.set_frames(self.frames)
        self.timing_stats.setText(exposure_summary(self.frames))
        self._set_loaded_state(True)
        self.canvas._fit_on_next = True
        self.goto(0)
        span = self.frames[-1].time_s - self.frames[0].time_s
        self.status.showMessage(
            f"{len(self.frames)} frames over {span:.3f} s "
            f"({format_time(self.frames[0].time_s)} to {format_time(self.frames[-1].time_s)})"
        )

    def _populate_strips(self):
        self.filmstrip.blockSignals(True)
        self.sheet.blockSignals(True)
        self.filmstrip.clear()
        self.sheet.clear()
        for f in self.frames:
            thumb = f.thumbnail(76)
            tag = f"{f.source_index}"
            if not f.is_new_drawing:
                tag += " hold"
            item = QListWidgetItem(QIcon(thumb), tag)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.filmstrip.addItem(item)

            big = f.thumbnail(118)
            label = (
                f"f{f.source_index}   {format_time(f.time_s)}\n"
                f"drawing {f.drawing_number}"
                + ("" if f.is_new_drawing else "  (hold)")
            )
            sitem = QListWidgetItem(QIcon(big), label)
            sitem.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.sheet.addItem(sitem)
        self.filmstrip.blockSignals(False)
        self.sheet.blockSignals(False)

    def _set_loaded_state(self, ready: bool):
        for w in (
            self.btn_first,
            self.btn_prev,
            self.btn_prev_draw,
            self.btn_play,
            self.btn_next,
            self.btn_next_draw,
            self.btn_last,
            self.view_toggle,
            self.fit_btn,
            self.hundred_btn,
        ):
            w.setEnabled(ready)

    # ---------------- navigation ----------------
    def goto(self, idx: int):
        if not self.frames:
            return
        self.current = max(0, min(len(self.frames) - 1, idx))
        self.refresh_frame()
        for widget in (self.filmstrip, self.sheet):
            widget.blockSignals(True)
            widget.setCurrentRow(self.current)
            widget.blockSignals(False)
        self.filmstrip.scrollToItem(
            self.filmstrip.item(self.current),
            QListWidget.ScrollHint.PositionAtCenter,
        )
        self.strip_bar.set_current(self.current)

    def step(self, delta: int):
        if not self.frames:
            return
        self.goto(self.current + delta)

    def next_drawing(self):
        for i in range(self.current + 1, len(self.frames)):
            if self.frames[i].is_new_drawing:
                self.goto(i)
                return
        self.goto(len(self.frames) - 1)

    def prev_drawing(self):
        for i in range(self.current - 1, -1, -1):
            if self.frames[i].is_new_drawing:
                self.goto(i)
                return
        self.goto(0)

    def _strip_selected(self, row: int):
        if row >= 0:
            self.goto(row)

    def _sheet_selected(self, row: int):
        if row >= 0:
            self.goto(row)

    def refresh_frame(self):
        if not self.frames:
            return
        f = self.frames[self.current]
        composed = compose(
            self.frames,
            self.current,
            self.mode_combo.currentText(),
            self.onion_depth.value(),
            self.onion_strength.value(),
            self.diff_gain.value(),
        )
        self.canvas.set_image(to_qimage(composed))

        self.lbl_pos.setText(f"{self.current + 1} of {len(self.frames)}")
        self.lbl_src.setText(str(f.source_index))
        self.lbl_tc.setText(format_time(f.time_s))
        self.lbl_drawing.setText(
            f"{f.drawing_number}" + ("" if f.is_new_drawing else "  (held)")
        )
        self.lbl_hold.setText(f"{f.hold_length} frame(s)")
        self.lbl_change.setText(f"{f.changed_fraction * 100:.3f} %")

    # ---------------- playback ----------------
    def toggle_play(self):
        if self.play_timer.isActive():
            self.stop_playback()
        else:
            if not self.frames:
                return
            self._retime_playback()
            self.play_timer.start()
            self.btn_play.setText("Pause")

    def stop_playback(self):
        self.play_timer.stop()
        self.btn_play.setText("Play")

    def _retime_playback(self):
        factors = {"1x": 1.0, "1/2x": 0.5, "1/4x": 0.25, "1/8x": 0.125, "2x": 2.0}
        speed = factors.get(self.speed_combo.currentText(), 1.0)
        fps = self.info.fps if self.info else 24.0
        interval = max(8, int(round(1000.0 / max(0.1, fps * speed))))
        self.play_timer.setInterval(interval)

    def _advance_playback(self):
        if not self.frames:
            self.stop_playback()
            return
        if self.current >= len(self.frames) - 1:
            if self.loop_check.isChecked():
                self.goto(0)
            else:
                self.stop_playback()
            return
        self.goto(self.current + 1)

    # ---------------- view switching ----------------
    def _toggle_view(self):
        self.pages.setCurrentIndex(1 if self.view_toggle.isChecked() else 0)
        self.view_toggle.setText(
            "Single frame" if self.view_toggle.isChecked() else "Contact sheet"
        )

    def _shortcut_toggle_view(self):
        self.view_toggle.setChecked(not self.view_toggle.isChecked())
        self._toggle_view()

    def _cycle_mode(self, target: str):
        current = self.mode_combo.currentText()
        self.mode_combo.setCurrentText("Normal" if current == target else target)

    # ---------------- timing ----------------
    def recompute_timing(self):
        if not self.frames:
            return
        analyse_exposure(self.frames, self.sensitivity.value())
        self.timing_stats.setText(exposure_summary(self.frames))
        self._populate_strips()
        self.strip_bar.set_frames(self.frames)
        self.goto(self.current)

    # ---------------- export ----------------
    def export_sequence(self):
        if not self.frames:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the PNG sequence", os.path.expanduser("~")
        )
        if not folder:
            return
        stem = os.path.splitext(os.path.basename(self.info.path))[0]
        width = max(4, len(str(self.frames[-1].source_index)))
        for f in self.frames:
            name = f"{stem}_f{f.source_index:0{width}d}.png"
            to_qimage(f.rgb).save(os.path.join(folder, name), "PNG")
        self.status.showMessage(f"Wrote {len(self.frames)} PNGs to {folder}")

    def export_contact_sheet(self):
        if not self.frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save the contact sheet",
            os.path.join(os.path.expanduser("~"), "contact_sheet.png"),
            "PNG (*.png)",
        )
        if not path:
            return

        cols = max(1, int(math.ceil(math.sqrt(len(self.frames) * 1.4))))
        rows = int(math.ceil(len(self.frames) / cols))
        tw, th, pad, caption = 260, 150, 10, 26
        cell_w, cell_h = tw + pad, th + caption + pad
        sheet = QImage(
            cols * cell_w + pad, rows * cell_h + pad, QImage.Format.Format_RGB888
        )
        sheet.fill(QColor("#101113"))

        p = QPainter(sheet)
        p.setFont(QFont("monospace", 9))
        for i, f in enumerate(self.frames):
            r, c = divmod(i, cols)
            x = pad + c * cell_w
            y = pad + r * cell_h
            img = to_qimage(f.rgb).scaled(
                tw,
                th,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.drawImage(x + (tw - img.width()) // 2, y, img)
            p.setPen(QColor(NEW_DRAWING if f.is_new_drawing else FG_MUTED))
            tag = "new" if f.is_new_drawing else "hold"
            p.drawText(
                x,
                y + th + 16,
                f"f{f.source_index}  {format_time(f.time_s)}  d{f.drawing_number} {tag}",
            )
        p.end()
        sheet.save(path, "PNG")
        self.status.showMessage(f"Contact sheet written to {path}")

    def export_csv(self):
        if not self.frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save the exposure sheet",
            os.path.join(os.path.expanduser("~"), "exposure_sheet.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "range_index",
                    "source_frame",
                    "timecode",
                    "seconds",
                    "drawing_number",
                    "new_drawing",
                    "hold_length",
                    "changed_pixels_pct",
                ]
            )
            for f in self.frames:
                w.writerow(
                    [
                        f.local_index,
                        f.source_index,
                        format_time(f.time_s),
                        f"{f.time_s:.6f}",
                        f.drawing_number,
                        int(f.is_new_drawing),
                        f.hold_length,
                        f"{f.changed_fraction * 100:.4f}",
                    ]
                )
        self.status.showMessage(f"Exposure sheet written to {path}")

    # ---------------- lifecycle ----------------
    def closeEvent(self, event):
        self.stop_playback()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        super().closeEvent(event)


def main():
    if av is None:
        print("PyAV is required. Install it with:  pip install av", file=sys.stderr)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setApplicationName("AnimStudy")
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    window = AnimStudy(initial)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
