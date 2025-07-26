# 유튜브 비디오에서 사람, 고양이, 개를 탐지하는 코드
# openCV와 YOLOv5 사용
# COCO 데이터셋

import cv2
import torch
import yt_dlp

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0, 15, 16] # 사람, 고양이, 개
class_names = model.names

# yt-dlp를 사용하여 유튜브 비디오 스트림을 가져오는 함수
def get_video_stream(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict['url']
        fps = info_dict.get('fps', 30)
    return video_url, fps


# # 문 개폐 로직을 처리하는 함수
# def door_control(labels):
#     """탐지된 객체 레이블을 기반으로 문 상태를 결정합니다."""
#     is_person_detected = 'person' in labels
#     is_animal_detected = 'cat' in labels or 'dog' in labels

#     if is_person_detected and not is_animal_detected:
#         return "Door: OPEN", (0, 255, 0)  # Green
#     elif is_animal_detected:
#         return "Door: DO NOT OPEN", (0, 0, 255)  # Red
#     else:
#         return "Door: CLOSED", (255, 255, 0)  # Cyan


# 비디오 스트림을 처리하고 객체 탐지 수행
def process_video(video_url, fps):
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("❌ Error: Cannot open video stream.")
        return

    # 👉 저장할 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱: .mp4
    out = cv2.VideoWriter('./week4/output/output_video.mp4', fourcc, fps, (640, 360))
    
    frame_delay = int(1000 / fps)
    frame_count = 0
    boxes = []

    # 🟡 영상 전체에서 cat 또는 dog이 나온 적이 있는지 추적할 변수
    video_had_animal = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360)) # 속도 향상을 위해 해상도 축소
        frame_count += 1
        
        detected_labels = []

        # 5프레임마다 객체 탐지 수행
        if frame_count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame, size=1280)
            boxes = results.xyxy[0].cpu().numpy()

        # 박스 표시
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            if conf < 0.35: # 너무 낮은 confidence는 무시
                continue
            
            # 탐지된 객체 레이블 추가
            label_name = class_names[int(cls)]
            detected_labels.append(label_name)
            
            # 박스와 레이블 그리기
            label = f"{class_names[int(cls)]} ({conf:.2f})"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 🟢 여기서 탐지 결과에 cat 또는 dog이 있으면 기록
        if 'cat' in detected_labels or 'dog' in detected_labels:
            video_had_animal = True
            
        # 👉 결과 프레임 저장
        out.write(frame)            
            
        # # 문 상태 결정
        # door_status, status_color = door_control(detected_labels)

        # # 화면에 문 상태 텍스트 표시
        # cv2.putText(frame, door_status, (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, status_color, 2)
        
        
        # 결과 프레임 표시
        cv2.imshow("YouTube Detection (person, cat, dog)", frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
    # 🔵 영상 끝난 후 최종 판단
    print("\n📢 [최종 판단]")
    if video_had_animal:
        print("🚫 영상 내에 고양이 또는 개가 탐지되었습니다. 문이 닫힙니다.")
    else:
        print("✅ 영상 내에 고양이 또는 개가 탐지되지 않았습니다. 문을 열어도 괜찮습니다.")




# 메인 함수
if __name__ == "__main__":
    # 사용자로부터 유튜브 URL 입력 받기
    youtube_url = input("유튜브 동영상 URL을 입력하세요: ").strip()
    
    if not youtube_url:
        print("❌ Error: 유효한 URL을 입력하세요.")
        exit(1)

    video_url, fps = get_video_stream(youtube_url)
    fps = 30 # 이미지에는 30으로 고정되어 있지만, get_video_stream에서 받은 fps를 사용하는 것이 일반적입니다.
    process_video(video_url, fps)