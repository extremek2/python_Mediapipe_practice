# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
from datetime import datetime
import time

import face_landmarker

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='data/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# STEP 4: 웃음 감지 설정
SMILE_THRESHOLD = 0.2  # 웃음 감지 임계값 (0.0~1.0)
CAPTURE_COOLDOWN = 2  # 촬영 간격 (초)
SHOW_LANDMARKS = False  # 랜드마크 표시 여부
last_capture_time = 0

# 사진 저장 폴더 생성
if not os.path.exists("smile_captures"):
    os.makedirs("smile_captures")


def detect_smile(blendshapes):
    """블렌드셰이프에서 웃음 정도 계산"""
    smile_score = 0.0

    # 웃음과 관련된 블렌드셰이프들
    smile_related = [
        'mouthSmileLeft',
        'mouthSmileRight',
        'cheekSquintLeft',
        'cheekSquintRight',
        'mouthUpperUpLeft',
        'mouthUpperUpRight'
    ]

    for blendshape in blendshapes:
        if blendshape.category_name in smile_related:
            smile_score += blendshape.score

    # 평균 계산
    return smile_score / len(smile_related) if smile_related else 0.0


def save_smile_photo(frame, smile_score):
    """웃는 사진 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"smile_captures/smile_{timestamp}_score_{smile_score:.2f}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 웃는 사진 저장: {filename} (웃음 점수: {smile_score:.2f})")
    return filename


print("웃음 감지 카메라가 시작되었습니다!")
print(f"웃음 임계값: {SMILE_THRESHOLD}")
print("'q': 종료, 's': 수동 촬영, '+/-': 임계값 조정, 'l': 랜드마크 토글")

# STEP 5: 실시간 처리 루프
while True:
    ret, frame = cap.read()

    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break

    # OpenCV BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    try:
        # 얼굴 랜드마크 및 블렌드셰이프 검출
        detection_result = detector.detect(mp_image)

        smile_score = 0.0
        smile_detected = False

        if detection_result.face_landmarks and detection_result.face_blendshapes:
            # 웃음 감지
            blendshapes = detection_result.face_blendshapes[0]
            smile_score = detect_smile(blendshapes)
            smile_detected = smile_score > SMILE_THRESHOLD

            # 랜드마크 그리기 (옵션에 따라)
            if SHOW_LANDMARKS:
                annotated_image = face_landmarker.draw_landmarks_on_image(rgb_frame, detection_result)
                display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            else:
                display_image = frame

            # 자동 촬영 (쿨다운 시간 체크)
            current_time = time.time()
            if smile_detected and (current_time - last_capture_time) > CAPTURE_COOLDOWN:
                save_smile_photo(display_image, smile_score)
                last_capture_time = current_time

                # 촬영 효과 (화면 깜빡임)
                flash_frame = np.ones_like(display_image) * 255
                cv2.imshow('Smile Detection Camera', flash_frame)
                cv2.waitKey(100)
        else:
            display_image = frame

    except Exception as e:
        print(f"검출 중 오류 발생: {e}")
        display_image = frame

    # UI 정보 표시
    info_y = 30

    # 웃음 점수 표시
    color = (0, 255, 0) if smile_detected else (0, 0, 255)
    cv2.putText(display_image, f"Smile Score: {smile_score:.2f}",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 임계값 표시
    cv2.putText(display_image, f"Threshold: {SMILE_THRESHOLD:.2f}",
                (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 웃음 감지 상태
    if smile_detected:
        cv2.putText(display_image, "SMILE DETECTED! 😊",
                    (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 조작 안내
    cv2.putText(display_image, "q:Quit | s:Manual | +/-:Threshold | l:Landmarks",
                (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 웃음 감지 진행바
    bar_width = 300
    bar_height = 20
    bar_x, bar_y = 10, info_y + 90

    # 배경
    cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    # 진행도
    progress_width = int(bar_width * min(smile_score / 1.0, 1.0))
    bar_color = (0, 255, 0) if smile_detected else (0, 255, 255)
    cv2.rectangle(display_image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), bar_color, -1)
    # 임계값 라인
    threshold_x = int(bar_x + bar_width * SMILE_THRESHOLD)
    cv2.line(display_image, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), (0, 0, 255), 2)

    cv2.imshow('Smile Detection Camera', display_image)

    # 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):  # 수동 촬영
        filename = save_smile_photo(display_image, smile_score)
    elif key == ord('+') or key == ord('='):  # 임계값 증가
        SMILE_THRESHOLD = min(1.0, SMILE_THRESHOLD + 0.1)
        print(f"웃음 임계값: {SMILE_THRESHOLD:.2f}")
    elif key == ord('-'):  # 임계값 감소
        SMILE_THRESHOLD = max(0.1, SMILE_THRESHOLD - 0.1)
        print(f"웃음 임계값: {SMILE_THRESHOLD:.2f}")
    elif key == ord('l'):  # 랜드마크 표시 토글
        SHOW_LANDMARKS = not SHOW_LANDMARKS
        status = "ON" if SHOW_LANDMARKS else "OFF"
        print(f"랜드마크 표시: {status}")

# STEP 6: 리소스 해제
cap.release()
cv2.destroyAllWindows()
print("프로그램이 종료되었습니다.")
print(f"촬영된 사진들은 'smile_captures' 폴더에 저장되어 있습니다.")