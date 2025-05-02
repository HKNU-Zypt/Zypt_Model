# FSTT_Model
# 모델 튜토리얼

## 모델 개요

- **모델 이름**: `facial_emotion_model.h5`
- **입력 데이터**: 얼굴 랜드마크 (468개의 (x, y) 좌표 → shape: `(468, 2)`)
- **입력 형식**: `numpy.ndarray` → shape: `(1, 468, 2)` (batch 포함)
- **출력 형식**: 소프트맥스 확률 (예: `[0.1, 0.7, 0.2]` 형태)
- **클래스 라벨**:
    
    ```
    concentration_labels
    0: focused
    1: not_focused
    2: drowsy
    ```
    

---

### 필요한 환경(내가 쓴거)

- Python ≥ 3.10
- TensorFlow ≥ 2.19
- Numpy ≥ 2.1.3
- Mediapipe ≥ 0.10.21

## mediapipe

- 설치 방법

```python
pip install mediapipe
```

https://github.com/google-ai-edge/mediapipe?tab=readme-ov-file

---

## 사용한 이미지 전처리 함수

### load_image_with_exif_orientation(image_path)

- 입력 데이터: 이미지 경로
- EXIF 회전 보정: 핸드폰 사진은 EXIF라는 이미지 회전 보정을 가진다. 만약 이 회전 보정을 고려하지 않는다면 스마트폰 사진에서 얼굴 랜드마크 추출을 실패할 수 있다.
- EXIF보정을 올바른 방향으로 전처리해주는 함수.

```python
import cv2
import numpy as np
import mediapipe as mp
import os
from PIL import Image, ExifTags

# Mediapipe 얼굴 랜드마크 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# EXIF 회전 보정 함수
def load_image_with_exif_orientation(image_path):
    image = Image.open(image_path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None and orientation in exif:
            orientation_value = exif[orientation]

            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except:
        pass

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
```

### resize_image(image, max_width=640)

- 입력 데이터: 이미지
- 이미지의 크기와 해상도를 조절해 데이터를 균일화 시키는 함수.

```python
# 이미지 리사이즈 함수
def resize_image(image, max_width=640):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image
```

### extract_face_landmarks(image_path, normazlize=True)

- 입력 데이터: 이미지 경로
- 얼굴의 랜드마크 추출 및 전처리를 포함하는 함수

```python
# 얼굴 랜드마크 추출 및 전처리 통합 함수
def extract_face_landmarks(image_path, normalize=True):
    # 1. 이미지 불러오기 (EXIF 보정 + 리사이즈)
    image = load_image_with_exif_orientation(image_path)
    image = resize_image(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Mediapipe FaceMesh 사용
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)

    # 3. 얼굴 랜드마크가 있으면 처리
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])  # 정규화된 좌표

        # 4. 선택적 정규화 (중심 정렬)
        if normalize:
            coords -= np.mean(coords, axis=0)

        return coords  # shape: (468, 2)

    return None  # 얼굴 없음
```

---

## 사용예시
```python
import numpy as np

# 이미지 경로 지정
image_path = "sample_images/user_photo.jpg"

# 얼굴 랜드마크 추출
landmarks = extract_face_landmarks(image_path, normalize=True)

if landmarks is not None:
    print("✔ 랜드마크 추출 완료!")
    print("shape:", landmarks.shape)  # 출력: (468, 2)
    print("예시 좌표:", landmarks[:5])  # 첫 5개 좌표 확인

    # 배치 차원 추가 (모델 입력용)
    input_data = np.expand_dims(landmarks, axis=0)  # shape: (1, 468, 2)

    # 모델 예측 (모델은 미리 로드되어 있어야 함)
    from tensorflow.keras.models import load_model
    model = load_model("concentration_model.h5")

    pred = model.predict(input_data)
    pred_label = np.argmax(pred)

    label_map = {0: "focused", 1: "not_focused", 2: "drowsy"}
    print("예측된 감정:", label_map[pred_label])

else:
    print("❌ 얼굴을 인식하지 못했습니다.")
```
## Notion
https://www.notion.so/1e78c93630e9803d9a57c0a1e2f3c8fe?pvs=4
