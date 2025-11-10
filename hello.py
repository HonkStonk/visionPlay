"""
PySide6 object-detection playground.

This starter script launches a dark-themed GUI that:
  * captures 720p video from the default webcam,
  * runs YOLOv8 object detection (Ultralytics) in a background thread, and
  * overlays the detections on the live feed with live stats.

Install dependencies (inside your virtual environment):
    pip install pyside6 opencv-python ultralytics

Run the app:
    python hello.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlopen

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QComboBox,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
)


@dataclass(frozen=True)
class ModelOption:
    label: str
    weights: str
    description: str
    download_url: Optional[str] = None


DEFAULT_MODEL = ModelOption(
    label="YOLOv8n (COCO general)",
    weights="yolov8n.pt",
    description="Fast, lightweight general-purpose detector.",
)

MODEL_OPTIONS = [
    DEFAULT_MODEL,
    ModelOption(
        label="YOLOv8s (COCO accuracy)",
        weights="yolov8s.pt",
        description="Higher accuracy COCO model, still realtime capable.",
    ),
    ModelOption(
        label="YOLOv8n-seg (Instance masks)",
        weights="yolov8n-seg.pt",
        description="Adds segmentation masks for each detected object.",
    ),
    ModelOption(
        label="YOLOv8n-pose (Pose & gestures)",
        weights="yolov8n-pose.pt",
        description="Keypoint detector for bodies/hands—great for gestures.",
    ),
    ModelOption(
        label="YOLOv8n-face (Face detector)",
        weights="yolov8n-face.pt",
        description="Face-focused detector with a single 'face' class.",
        download_url="https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
    ),
    ModelOption(
        label="YOLOv8n-obb (Oriented boxes)",
        weights="yolov8n-obb.pt",
        description="Detects objects with rotated boxes (drones, aerial).",
    ),
]


def ensure_weights(option: ModelOption) -> str:
    """Return a path Ultralytics can load, downloading custom weights if needed."""
    if not option.download_url:
        return option.weights

    weights_dir = Path("models")
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_path = weights_dir / Path(option.weights).name
    if target_path.exists():
        return str(target_path)

    try:
        with urlopen(option.download_url) as response:
            data = response.read()
    except Exception as exc:  # pragma: no cover - feedback path
        raise RuntimeError(f"Download failed for {option.label}: {exc}") from exc

    target_path.write_bytes(data)
    return str(target_path)


@dataclass
class DetectionStats:
    fps: float
    device: str
    counts: Dict[str, int]
    model_name: str


class DetectionWorker(QThread):
    """Background thread that grabs frames, runs YOLOv8, and emits annotated frames."""

    frame_ready = Signal(object)
    stats_ready = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        source: int = 0,
        frame_size: Tuple[int, int] = (1280, 720),
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        model_option: ModelOption = DEFAULT_MODEL,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._frame_size = frame_size
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._model_option = model_option
        self._running = False

    def stop(self) -> None:
        """Request the worker to halt and wait for it to finish."""

        self._running = False
        self.wait()

    def run(self) -> None:
        try:
            from ultralytics import (
                YOLO,
            )  # Local import keeps GUI responsive if missing.
            import torch
        except Exception as exc:  # pragma: no cover - feedback path
            self.error.emit(f"Import error: {exc}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            weights_path = ensure_weights(self._model_option)
            model = YOLO(weights_path)
            model.to(device)
        except Exception as exc:  # pragma: no cover - feedback path
            self.error.emit(f"Model load failed: {exc}")
            return

        cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            self.error.emit("Unable to open webcam. Check device index / permissions.")
            return

        self._running = True
        last_time = time.perf_counter()
        frame_counter = 0

        try:
            while self._running:
                grabbed, frame = cap.read()
                if not grabbed:
                    self.error.emit("Failed to read frame from webcam.")
                    break

                if self._frame_size:
                    frame = cv2.resize(frame, self._frame_size)

                try:
                    results = model.predict(
                        frame,
                        imgsz=640,
                        conf=self._conf,
                        iou=self._iou,
                        verbose=False,
                        device=device,
                    )
                except Exception as exc:  # pragma: no cover - feedback path
                    self.error.emit(f"Inference error: {exc}")
                    break

                if not results:
                    continue

                result = results[0]
                annotated = result.plot()

                labels = [
                    model.names[int(cls_id)]
                    for cls_id in (result.boxes.cls.tolist() if result.boxes else [])
                ]
                counts = Counter(labels)

                frame_counter += 1
                now = time.perf_counter()
                elapsed = now - last_time
                fps = frame_counter / elapsed if elapsed > 0 else 0.0

                stats = DetectionStats(
                    fps=fps,
                    device=device,
                    counts=dict(counts),
                    model_name=self._model_option.label,
                )

                self.frame_ready.emit(annotated)
                self.stats_ready.emit(stats)
        finally:
            cap.release()


class VideoWidget(QLabel):
    """Label-based widget that preserves aspect ratio and shows frames."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)

    def show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(image.copy()))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VisionPlay | YOLOv8 Live Detection")
        self.resize(1320, 840)

        self._video_widget = VideoWidget()
        self._status_label = QLabel("Initializing.")
        self._model_options = MODEL_OPTIONS
        self._current_model = DEFAULT_MODEL
        self._model_combo: Optional[QComboBox] = None
        self._worker: Optional[DetectionWorker] = None
        self._status_initialized = False

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._video_widget)

        info_bar = QHBoxLayout()
        info_bar.setSpacing(16)
        info_bar.addWidget(self._status_label, stretch=1)
        layout.addLayout(info_bar)

        self.setCentralWidget(central)
        self._build_toolbar()
        self._build_statusbar()
        self._start_worker(self._current_model)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Session Controls")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        toolbar.addSeparator()

        model_label = QLabel("Model:")
        model_label.setContentsMargins(6, 0, 4, 0)
        toolbar.addWidget(model_label)

        combo = QComboBox()
        for option in self._model_options:
            combo.addItem(option.label)
            index = combo.count() - 1
            combo.setItemData(index, option.description, Qt.ItemDataRole.ToolTipRole)

        combo.currentIndexChanged.connect(self._handle_model_change)
        combo.setMaximumWidth(260)
        combo.setToolTip("Select which YOLOv8 variant to run")
        self._model_combo = combo
        toolbar.addWidget(combo)
        combo.setCurrentIndex(self._model_options.index(self._current_model))

    def _build_statusbar(self) -> None:
        status = QStatusBar()
        status.showMessage("Starting camera.")
        self.setStatusBar(status)

    def _start_worker(self, model_option: ModelOption) -> None:
        self._current_model = model_option
        self._status_initialized = False
        if self.statusBar():
            self.statusBar().showMessage("Starting camera.")
        self._status_label.setText(f"Loading {model_option.label}...")

        worker = DetectionWorker(model_option=model_option)
        worker.frame_ready.connect(self._video_widget.show_frame)
        worker.stats_ready.connect(self._update_stats)
        worker.error.connect(self._handle_error)
        worker.start()
        self._worker = worker

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.deleteLater()
            self._worker = None

    def _handle_model_change(self, index: int) -> None:
        if index < 0 or index >= len(self._model_options):
            return
        next_model = self._model_options[index]
        if next_model == self._current_model:
            return
        self._status_label.setText(f"Switching to {next_model.label}...")
        self._stop_worker()
        self._start_worker(next_model)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_worker()
        super().closeEvent(event)

    def _update_stats(self, stats: DetectionStats) -> None:
        counts_text = ", ".join(
            f"{label}: {count}" for label, count in stats.counts.items()
        )
        counts_text = counts_text or "No detections"
        self._status_label.setText(
            f"{stats.model_name} | FPS: {stats.fps:0.1f} | Device: {stats.device} | {counts_text}"
        )
        if not self._status_initialized:
            self.statusBar().showMessage(
                f"Live detection running ({stats.model_name})."
            )
            self._status_initialized = True

    def _handle_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Error: {message}")
        self._status_label.setText(message)


def apply_dark_palette(app: QApplication) -> None:
    """Set a Fusion dark theme to match Windows dark mode."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QToolBar { spacing: 6px; }
        QLabel { font-size: 14px; }
        """
    )


def main() -> int:
    app = QApplication(sys.argv)
    apply_dark_palette(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
