import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QHBoxLayout,
                             QVBoxLayout, QPushButton, QSlider, QMainWindow)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CAPTURE_FOLDER = "smile_captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------
def detect_smile(blendshapes):
    """블렌드셰이프에서 웃음 점수 계산"""
    smile_score = 0.0
    smile_related = [
        'mouthSmileLeft', 'mouthSmileRight',
        'cheekSquintLeft', 'cheekSquintRight',
        'mouthUpperUpLeft', 'mouthUpperUpRight'
    ]
    for b in blendshapes:
        if b.category_name in smile_related:
            smile_score += b.score
    return smile_score / len(smile_related) if smile_related else 0.0

def save_smile_photo(frame, smile_score):
    """웃음 사진 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CAPTURE_FOLDER}/smile_{timestamp}_score_{smile_score:.2f}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def draw_landmarks_on_image(frame, result):
    """랜드마크 표시"""
    annotated = frame.copy()
    for lm in result.face_landmarks[0].landmark:
        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
    return annotated

# ---------------------------
# Video Thread
# ---------------------------
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    captured_signal = pyqtSignal(str)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self._run_flag = True
        self.SMILE_THRESHOLD = 0.2
        self.CAPTURE_COOLDOWN = 2
        self.last_capture_time = 0
        self.show_landmarks = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            smile_score = 0.0
            smile_detected = False

            try:
                result = self.detector.detect(mp_image)
                if result.face_landmarks and result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    smile_score = detect_smile(blendshapes)
                    smile_detected = smile_score > self.SMILE_THRESHOLD

                    current_time = time.time()
                    display_frame = frame.copy()
                    if self.show_landmarks:
                        display_frame = draw_landmarks_on_image(display_frame, result)

                    # 자동 캡처
                    if smile_detected and (current_time - self.last_capture_time) > self.CAPTURE_COOLDOWN:
                        filename = save_smile_photo(display_frame, smile_score)
                        self.captured_signal.emit(filename)
                        self.last_capture_time = current_time

                else:
                    display_frame = frame.copy()

            except Exception as e:
                display_frame = frame.copy()

            # 진행도 바
            bar_width, bar_height = 300, 20
            bar_x, bar_y = 10, 10
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x+bar_width, bar_y+bar_height), (50,50,50), -1)
            progress_width = int(bar_width * min(smile_score / 1.0, 1.0))
            bar_color = (0,255,0) if smile_detected else (0,255,255)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x+progress_width, bar_y+bar_height), bar_color, -1)
            threshold_x = int(bar_x + bar_width * self.SMILE_THRESHOLD)
            cv2.line(display_frame, (threshold_x, bar_y), (threshold_x, bar_y+bar_height), (0,0,255), 2)

            # 웃음 점수 표시
            color = (0,255,0) if smile_detected else (0,0,255)
            cv2.putText(display_frame, f"Smile Score: {smile_score:.2f}", (10, bar_y+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            self.change_pixmap_signal.emit(display_frame)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ---------------------------
# Main GUI
# ---------------------------
class SmileDetectionApp(QMainWindow):
    def __init__(self, detector):
        super().__init__()
        self.setWindowTitle("Smile Detection App")
        self.detector = detector

        # 중앙 위젯 & 레이아웃
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # 왼쪽: 실시간 영상
        left_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # 컨트롤 버튼
        self.btn_capture = QPushButton("Manual Capture")
        self.btn_capture.clicked.connect(self.manual_capture)
        left_layout.addWidget(self.btn_capture)

        self.btn_landmark = QPushButton("Toggle Landmarks")
        self.btn_landmark.clicked.connect(self.toggle_landmarks)
        left_layout.addWidget(self.btn_landmark)

        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(1, 100)
        self.slider_threshold.setValue(int(0.2*100))
        self.slider_threshold.valueChanged.connect(self.update_threshold)
        left_layout.addWidget(self.slider_threshold)

        layout.addLayout(left_layout)

        # 오른쪽: 마지막 캡처 사진
        right_layout = QVBoxLayout()
        self.capture_label = QLabel()
        self.capture_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.capture_label)
        layout.addLayout(right_layout)

        # 비디오 스레드 시작
        self.thread = VideoThread(detector)
        self.thread.change_pixmap_signal.connect(self.update_video)
        self.thread.captured_signal.connect(self.update_captured_photo)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def update_video(self, frame):
        qimg = self.convert_cv_qt(frame)
        self.video_label.setPixmap(qimg)

    def update_captured_photo(self, filepath):
        pixmap = QPixmap(filepath)
        pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio)
        self.capture_label.setPixmap(pixmap)

    def manual_capture(self):
        frame = self.thread.change_pixmap_signal  # 마지막 프레임 접근 방법에 따라 조정 가능
        # 여기서는 스레드 내부에서 emit된 frame 저장하는 방식 필요
        # 예제에서는 생략, 필요시 frame 저장 변수 추가

    def toggle_landmarks(self):
        self.thread.show_landmarks = not self.thread.show_landmarks

    def update_threshold(self, value):
        self.thread.SMILE_THRESHOLD = value / 100.0

    @staticmethod
    def convert_cv_qt(frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


# ---------------------------
# 실행
# ---------------------------
if __name__ == "__main__":
    base_options = python.BaseOptions(model_asset_path='data/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    app = QApplication(sys.argv)
    window = SmileDetectionApp(detector)
    window.show()
    sys.exit(app.exec_())
