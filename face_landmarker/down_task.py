# face_landmarker task 파일을 다운로드 받기 위해 1번만 실행

import os
import urllib.request

# Directory 경로 설정
package_dir = os.path.dirname(__file__)
data_dir = os.path.join(package_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# File 이름 및 경로 설정
filename = "face_landmarker_v2_with_blendshapes.task"
filepath = os.path.join(data_dir, filename)

# 소스 url 주소
url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# 해당 파일 없을 경우 다운로드
if not os.path.exists(filepath):
    print(f"{filename}' 이 없으므로 다운로드를 진행합니다.")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"{filepath}에 다운로드가 완료되었습니다.")
    except Exception as e:
        print(f"다운로드 에러: {e}")
