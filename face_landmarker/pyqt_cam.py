import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QHBoxLayout,
                             QVBoxLayout, QPushButton, QSlider, QMainWindow, QFrame)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CAPTURE_FOLDER}/smile_{timestamp}_score_{int(smile_score*100)}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def draw_landmarks_on_image(frame, result):
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
    captured_signal = pyqtSignal(object, float)  # (frame_or_filepath, score)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self._run_flag = True
        self.SMILE_THRESHOLD = 0.2
        self.CAPTURE_COOLDOWN = 2
        self.last_capture_time = 0
        self.show_landmarks = False
        self.auto_save = True

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
                display_frame = frame.copy()

                if result.face_landmarks and result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    smile_score = detect_smile(blendshapes)
                    smile_detected = smile_score > self.SMILE_THRESHOLD

                    if self.show_landmarks:
                        display_frame = draw_landmarks_on_image(display_frame, result)

                    current_time = time.time()
                    if smile_detected and (current_time - self.last_capture_time) > self.CAPTURE_COOLDOWN:
                        if self.auto_save:
                            filename = save_smile_photo(display_frame, smile_score)
                            self.captured_signal.emit(filename, smile_score)
                        else:
                            self.captured_signal.emit(display_frame, smile_score)
                        self.last_capture_time = current_time

                # 진행도 바 표시
                bar_w, bar_h = 300, 20
                cv2.rectangle(display_frame, (10,10), (10+bar_w,10+bar_h), (50,50,50), -1)
                prog = int(bar_w * min(smile_score,1.0))
                color = (0,255,0) if smile_detected else (0,255,255)
                cv2.rectangle(display_frame, (10,10), (10+prog,10+bar_h), color, -1)
                thresh_x = int(10 + bar_w*self.SMILE_THRESHOLD)
                cv2.line(display_frame, (thresh_x,10), (thresh_x,10+bar_h), (0,0,255), 2)
                col = (0,255,0) if smile_detected else (0,0,255)
                cv2.putText(display_frame, f"Smile: {int(smile_score*100)}%", (10,45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            except Exception as e:
                display_frame = frame.copy()

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
        # QMainWindow 초기화
        self.setWindowTitle("Smile Detection App")

        # 캠, 프리뷰 사이즈
        cam_width, cam_height = 640, 480
        preview_width, preview_height = 640, 480

        # 레이아웃 여백 고려 (간단히 padding 20px 정도)
        padding = 20

        window_width = cam_width + preview_width + padding
        window_height = max(cam_height, preview_height) + padding

        self.resize(window_width, window_height)

        self.detector = detector

        self.pending_frame = None
        self.pending_score = None


        # ----------------------
        # 왼쪽 프레임 (캠)
        # ----------------------
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Box)  # 경계선
        left_layout = QVBoxLayout(left_frame)

        # 왼쪽 상단 고정 레이블
        self.lbl_camera = QLabel("실시간 영상")
        self.lbl_camera.setAlignment(Qt.AlignLeft)
        left_layout.addWidget(self.lbl_camera)

        # 왼쪽 중앙: 카메라 프레임
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignLeft)
        left_layout.addWidget(self.video_label)

        # 슬라이더 레이블
        self.lbl_threshold = QLabel("웃음 강도 조절")
        left_layout.addWidget(self.lbl_threshold)

        # 슬라이더
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(1,100)
        self.slider_thresh.setValue(int(0.2*100))
        self.slider_thresh.valueChanged.connect(self.update_threshold)
        left_layout.addWidget(self.slider_thresh)


        # ----------------------
        # 오른쪽 프레임 (프리뷰)
        # ----------------------

        # 오른쪽 프레임 초기화
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.Box)  # 경계선
        right_frame.setFixedSize(640, 480)
        right_layout = QVBoxLayout(right_frame)

        # 오른쪽 고정 레이블
        self.preview_label = QLabel("최신 웃음 사진")
        self.preview_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.preview_label)

        # 오른쪽: 캡처 프리뷰
        self.capture_label = QLabel("No preview")
        self.capture_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.capture_label)

        # 오른쪽: 저장 버튼 (자동 저장 모드 OFF 시 활성화)
        self.btn_save = QPushButton("저장")
        self.btn_save.clicked.connect(self.save_preview)
        self.btn_save.setVisible(False)
        right_layout.addWidget(self.btn_save)

        # 오른쪽: 취소 버튼 (자동 저장 모드 OFF 시 활성화)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.clicked.connect(self.cancel_preview)
        self.btn_cancel.setVisible(False)
        right_layout.addWidget(self.btn_cancel)

        # 오른쪽: 자동 저장 모드 (ON/OFF)
        self.btn_auto = QPushButton("자동 저장 모드: ON")
        self.btn_auto.clicked.connect(self.toggle_auto_save)
        right_layout.addWidget(self.btn_auto)


        # ----------------------
        # 메인 레이아웃
        # ----------------------
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_frame)
        main_layout.addWidget(right_frame)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

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

    def update_captured_photo(self, data, score):
        if isinstance(data, str):  # Auto Save ON
            pixmap = QPixmap(data).scaled(640,480,Qt.KeepAspectRatio)
            self.capture_label.setPixmap(pixmap)
            self.pending_frame = None
            self.pending_score = None
        else:  # Auto Save OFF
            self.pending_frame = data
            self.pending_score = score

            # 점수 오버레이
            overlay = data.copy()
            text = f"😊 {int(score*100)}%"
            cv2.putText(overlay, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 2, cv2.LINE_AA)
            qpix = self.convert_cv_qt(overlay).scaled(640,480,Qt.KeepAspectRatio)
            self.capture_label.setPixmap(qpix)

    def save_preview(self):
        if self.pending_frame is not None and self.pending_score is not None:
            filename = save_smile_photo(self.pending_frame, self.pending_score)
            print(f"Saved: {filename}")
            self.pending_frame = None
            self.pending_score = None

    def cancel_preview(self):
        self.capture_label.setText("No preview")
        self.pending_frame = None
        self.pending_score = None

    def manual_capture(self):
        if self.thread._run_flag:
            frame = self.thread.change_pixmap_signal  # 참고: 필요 시 마지막 프레임 변수로 저장
            # 간단히 OFF 모드와 동일하게 처리 가능
            pass

    def toggle_landmarks(self):
        self.thread.show_landmarks = not self.thread.show_landmarks

    def toggle_auto_save(self):
        self.thread.auto_save = not self.thread.auto_save
        status = "ON" if self.thread.auto_save else "OFF"
        self.btn_auto.setText(f"Auto Save: {status}")

        # Save / Cancel 버튼 표시 여부
        show_buttons = not self.thread.auto_save
        self.btn_save.setVisible(show_buttons)
        self.btn_cancel.setVisible(show_buttons)

    def update_threshold(self, value):
        self.thread.SMILE_THRESHOLD = value / 100.0

    @staticmethod
    def convert_cv_qt(frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch*w
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
