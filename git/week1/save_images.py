# save_images.py
from datasets import load_dataset
import os

# Hugging Face 데이터셋 로드 (D드라이브에 캐시 저장)
print("데이터셋 로딩 중...")
ds = load_dataset("ethz/food101", split='train', cache_dir="D:/food101_dataset")

# 이미지를 저장할 폴더 경로
SAVE_DIR = "D:/my_cv_project/original_images"

# 저장할 이미지 개수
NUM_TO_SAVE = 100 # 전처리를 위해 넉넉하게 100장 저장

# 폴더가 없으면 생성
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"{NUM_TO_SAVE}개의 이미지 저장을 시작합니다...")
for i, example in enumerate(ds):
    if i >= NUM_TO_SAVE:
        break
    
    image = example['image']
    label_name = ds.features['label'].int2str(example['label'])
    
    # 파일 이름 형식: label_index.png (예: apple_pie_0.png)
    filename = f"{label_name}_{i}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    
    # 이미지 저장
    image.save(save_path)

    if (i + 1) % 10 == 0:
        print(f"{i + 1}개 이미지 저장 완료...")

print("\n이미지 저장이 모두 완료되었습니다.")
print(f"저장된 위치: {SAVE_DIR}")