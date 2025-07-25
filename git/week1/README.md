# 전처리 과정 설명


이 프로젝트는 Hugging Face 데이터셋에서 이미지를 가져와 AI 학습을 위한 전처리를 수행하는 `image_preprocessing.py` 코드의 설명입니다. 

원본 데이터 출처: <https://huggingface.co/datasets/ethz/food101>



### 1. 기본 전처리 (기본 문제)

-   **크기 조절**: 모든 이미지를 `224x224` 픽셀의 동일한 크기로 통일
-   **색상 변환**: 컬러 이미지를 **Grayscale(흑백)** 이미지로 변환
-   **노이즈 제거**: **가우시안 블러(Gaussian Blur)** 필터를 적용하여 이미지의 미세한 노이즈를 부드럽게 처리
-   **정규화(Normalization)**: 픽셀 값을 0~255 범위에서 0.0~1.0 범위로 변환
-   **데이터 증강**: 좌우 반전(Flip), 회전(Rotate), 색상 변화


###  2. 이상치 탐지 및 필터링 (심화 문제)

-   **너무 어두운 이미지 제거**: 이미지 전체 픽셀의 평균 밝기를 계산하여, 설정된 임계값(`BRIGHTNESS_THRESHOLD`)보다 낮으면 학습 데이터에서 제외
-   **객체가 너무 작은 이미지 제거**: 이미지에서 가장 면적이 넓은 객체(Contour)를 찾고, 이 객체가 전체 이미지에서 차지하는 비율을 계산. 이 비율이 설정된 임계값(`OBJECT_SIZE_THRESHOLD`)보다 작으면 제외

  

## 실행 방법

1.  'save_images.py' 파일 실행>> `D:\my_cv_project\original_images` 폴더에 100개의 원본 이미지 저장
2.  `image_preprocessing.py` 파일 실행>> 전처리!
3.  'preprocessed_samples\`>> 폴더 안에 전처리 및 증강이 완료된 샘플 이미지들이 저장됨!
