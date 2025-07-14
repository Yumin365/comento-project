import cv2
import numpy as np
import os
import glob


# --- 설정 (사용자 환경에 맞게 수정하세요) ---

# 1. 원본 이미지가 있는 폴더 경로
# D드라이브의 'my_cv_project/original_images' 폴더로 지정
INPUT_DIR = "D:/my_cv_project/original_images"

# 2. 전처리된 이미지를 저장할 폴더 경로
OUTPUT_DIR = "./preprocessed_samples/"

# 3. 저장할 샘플 이미지 개수
NUM_SAMPLES_TO_SAVE = 5

# --- 심화 문제에 필요한 설정값 ---

# 4. 이상치 탐지: 너무 어두운 이미지를 거르기 위한 밝기 임계값 (0~255 사이)
# 값이 낮을수록 더 어두운 이미지까지 허용
BRIGHTNESS_THRESHOLD = 50

# 5. 이상치 탐지: 객체 크기가 너무 작은 이미지를 거르기 위한 비율 임계값 (0.0 ~ 1.0 사이)
# 전체 이미지 면적 대비 가장 큰 객체의 면적 비율
# 값이 낮을수록 더 작은 객체까지 허용
OBJECT_SIZE_THRESHOLD = 0.05

# --- 전처리 및 증강 함수 ---

def preprocess_image(image_path):
    """이미지 하나를 읽고 전처리 및 증강을 수행하는 함수"""
    
    # 1. 이미지 읽기
    # 한글 경로 문제 방지를 위해 np.fromfile과 cv2.imdecode 사용
    try:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return None
    except Exception as e:
        print(f"이미지 로딩 중 에러 발생: {image_path}, 에러: {e}")
        return None

    # --- [심화 문제] 이상치 탐지 및 필터링 ---
    
    # 1. 너무 어두운 이미지 제거 (평균 밝기 기준)
    gray_for_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_for_check)
    if avg_brightness < BRIGHTNESS_THRESHOLD:
        print(f"[{os.path.basename(image_path)}] 너무 어두워서 제외 (밝기: {avg_brightness:.2f})")
        return None

    # 2. 객체 크기가 너무 작은 이미지 제거
    # 이미지에서 가장 큰 contour(윤곽선)의 면적을 객체 크기로 간주
    blurred = cv2.GaussianBlur(gray_for_check, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        object_area = cv2.contourArea(max_contour)
        total_area = img.shape[0] * img.shape[1]
        object_ratio = object_area / total_area
        
        if object_ratio < OBJECT_SIZE_THRESHOLD:
            print(f"[{os.path.basename(image_path)}] 객체가 너무 작아서 제외 (비율: {object_ratio:.2f})")
            return None
    else: # Contour가 아예 없는 경우 (예: 완전한 백색 이미지)
        print(f"[{os.path.basename(image_path)}] 객체를 찾을 수 없어 제외")
        return None

    # --- [기본 문제] 전처리 ---
    
    # 1. 크기 조절 (224x224)
    resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # 2. 색상 변환 (Grayscale)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # 3. 노이즈 제거 (Blur 필터 적용)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 4. 정규화 (Normalize) - 픽셀 값을 0~1 사이로
    normalized_img = blurred_img / 255.0

    # --- [기본 문제] 데이터 증강 (Data Augmentation) ---
    
    augmented_images = {}
    
    # 원본(전처리만 된) 이미지
    # 저장을 위해 float 타입인 normalized_img를 8-bit 정수형(0-255)으로 다시 변환
    augmented_images['preprocessed'] = np.uint8(normalized_img * 255.0)

    # 1. 좌우 반전
    augmented_images['flipped'] = cv2.flip(augmented_images['preprocessed'], 1)

    # 2. 회전 (15도)
    h, w = augmented_images['preprocessed'].shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1) # 중심점, 각도, 스케일
    augmented_images['rotated'] = cv2.warpAffine(augmented_images['preprocessed'], matrix, (w, h))

    # 3. 색상 변화 (밝기 조절로 대체)
    # Grayscale이라 색상 변화 대신 밝기 조절을 수행
    # 30만큼 밝게 만듦 (값이 255를 넘지 않도록 np.clip 사용)
    brighter_img = np.clip(augmented_images['preprocessed'].astype(np.int16) + 30, 0, 255).astype(np.uint8)
    augmented_images['brighter'] = brighter_img

    return augmented_images


# --- 메인 코드 실행 부분 ---

if __name__ == "__main__":
    
    # 결과 저장 폴더가 없으면 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"'{OUTPUT_DIR}' 폴더 생성 완료.")

    # 원본 이미지 폴더에서 모든 이미지 파일 경로 가져오기
    # jpg, png, bmp 등 다양한 확장자 지원
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    print(f"총 {len(image_paths)}개의 원본 이미지 발견.")
    
    saved_count = 0
    # 모든 이미지에 대해 전처리 수행
    for path in image_paths:
        if saved_count >= NUM_SAMPLES_TO_SAVE:
            print(f"\n목표한 샘플 {NUM_SAMPLES_TO_SAVE}개를 모두 저장했습니다.")
            break

        # 전처리 및 증강 함수 호출
        augmented_results = preprocess_image(path)

        # 함수가 None을 반환하면 (이상치로 필터링되면) 건너뜀
        if augmented_results is None:
            continue

        # 결과 저장
        base_filename = os.path.splitext(os.path.basename(path))[0]
        for key, img_data in augmented_results.items():
            if saved_count < NUM_SAMPLES_TO_SAVE:
                # 저장할 파일 이름: 원본이름_증강타입.png (예: food_01_flipped.png)
                output_filename = f"{base_filename}_{key}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # 한글 경로 저장을 위해 imencode 사용
                is_success, im_buf_arr = cv2.imencode(".png", img_data)
                if is_success:
                    im_buf_arr.tofile(output_path)
                    print(f" -> 저장 완료: {output_filename}")
                    saved_count += 1
                else:
                    print(f"이미지 저장 실패: {output_filename}")
            else:
                break
    
    if saved_count < NUM_SAMPLES_TO_SAVE:
        print(f"\n작업 완료. 총 {saved_count}개의 이미지를 저장했습니다.")