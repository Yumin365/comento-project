import pytest


#  #### 기본적인 DepthMap 생성코드 (OpenCV 활용)
# import cv2
# import numpy as np

#  # 이미지 로드
# image= cv2.imread('sample2.jpg')

#  # 그레이스케일 변환
# gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#  # 깊이 맵 생성(가상의 깊이 적용)
# depth_map= cv2.applyColorMap(gray, cv2.COLORMAP_JET)

#  # 결과출력
# cv2.imshow('Original Image', image)
# cv2.imshow('DepthMap', depth_map) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()




#--------------------------------------


#  ####심화코드: DepthMap을기반으로3D 포인트클라우드생성
#  #실제 3D 스캐닝 데이터는 아니지만, 
#  # 2D 이미지의 밝기 정보를 3D 공간의 깊이 정보로 '가정'하여
#  # 점들의 집합(포인트 클라우드)을 만드는 개념을 보여줍니다
 
 
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt # Matplotlib 임포트
# from mpl_toolkits.mplot3d import Axes3D # 3D 플롯을 위해 필요


# # 이미지 로드
# image= cv2.imread('./week2/sample5.jpg')

# # 그레이스케일 변환
# gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # DepthMapb생성
# depth_map= cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# # 3D 포인트 클라우드 변환
# h, w= depth_map.shape[:2]
# X, Y= np.meshgrid(np.arange(w), np.arange(h))
# # np.arange(w): 0부터 w-1까지의 정수 배열
# # np.arange(h): 0부터 h-1까지의 정수 배열
# # np.meshgrid: 2D 그리드 좌표 생성
# Z= gray.astype(np.float32)  # Depth값(이미지의 밝기값)을 Z축으로 사용
# # 밝기가 클수록 가까이 있는 것으로 가정

# # 3D 좌표생성
# points_3d = np.dstack((X, Y, Z))
# # 배열을 Z축으로 쌓아 3D 좌표 생성


# # # 결과출력
# # cv2.imshow('DepthMap', depth_map)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()




# # ----------------- 3D 포인트 클라우드 시각화 (Matplotlib) -----------------

# # 3D 좌표생성
# # X, Y, Z 배열을 (픽셀 수, 1) 형태로 평탄화한 후, 다시 (픽셀 수, 3)으로 합칩니다.
# # Matplotlib의 scatter는 1D 배열을 입력으로 받기 때문에 reshape(-1)을 사용합니다.
# points_3d_flattened = np.dstack((X, Y, Z)).reshape(-1, 3) # 모든 점을 하나의 리스트로 만듭니다.

# fig = plt.figure(figsize=(10, 8)) # 새 그림(figure) 객체 생성
# ax = fig.add_subplot(111, projection='3d') # 3D 서브플롯 추가

# # 산점도(scatter plot) 그리기
# # X, Y, Z 좌표를 각각 첫 번째, 두 번째, 세 번째 열에서 가져옵니다.
# # s=1은 점의 크기, alpha=0.5는 투명도를 나타냅니다.
# # c는 점의 색상인데, 여기서는 Z값을 색상으로 사용하여 깊이에 따라 색이 변하도록 합니다.
# # cmap은 컬러맵을 지정합니다.
# sc = ax.scatter(points_3d_flattened[:, 0], -points_3d_flattened[:, 1], points_3d_flattened[:, 2],
#                 c=points_3d_flattened[:, 2], cmap='jet', s=1, alpha=0.8)

# ax.set_xlabel('X-axis (Width)')
# ax.set_ylabel('Y-axis (Height)')
# ax.set_zlabel('Z-axis (Depth/Brightness)')
# ax.set_title('3D Point Cloud from Image Brightness')
# fig.colorbar(sc, shrink=0.5, aspect=5, label='Depth Value (Brightness)') # 컬러바 추가

# plt.savefig('.\week2\output_3d_plot_4') # 3D 플롯을 파일로 저장

# plt.show() # 3D 플롯 보여주기

# # -------------------------------------------------------------------------

# # 결과출력 (기존 DepthMap 시각화)
# cv2.imshow('Original Image', image) # 원본 이미지 추가
# cv2.imshow('DepthMap', depth_map)
# cv2.imwrite('.\week2\output_depth_map_4.jpg', depth_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-------------------------------------
# # 테스트 코드
# def test_generate_depth_map():
    
#     assert depth_map.shape== image.shape, "출력 크기가 입력 크기와 다릅니다." # assert: 조건이 True가 아니면 AssertionError 발생
#     assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."
 
 
 
#  # pytest실행
# if __name__ == "__main__":
#     pytest.main() # test_로 시작하는 함수들을 자동으로 찾아서 실행


import cv2
import numpy as np
import open3d as o3d # Open3D 임포트

# 이미지 로드
image = cv2.imread('.\week2\sample2.jpg')

# 이미지가 제대로 로드되었는지 확인
if image is None:
    print("이미지를 로드할 수 없습니다. 'sample2.jpg' 파일이 존재하고 올바른 이미지인지 확인하세요.")
    exit()

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3D 포인트 클라우드 변환
h, w = gray.shape[:2]
X, Y = np.meshgrid(np.arange(w), -np.arange(h))
Z = gray.astype(np.float32)

# 3D 좌표생성 (Open3D는 (N, 3) 형태의 NumPy 배열을 선호)
points_3d_flattened = np.dstack((X, Y, Z)).reshape(-1, 3)

# ----------------- 3D 포인트 클라우드 시각화 (Open3D) -----------------
# 1. Open3D PointCloud 객체 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d_flattened)

# 2. 색상 정보 추가 
# 원본 이미지의 색상을 각 3D 포인트에 매핑
# 이를 위해 원본 이미지도 224x224 (또는 gray 이미지와 동일한 크기)로 리사이징해야 함!
# 현재 예시에서는 gray 이미지의 밝기를 Z값으로 썼으므로, 
# 원본 BGR 이미지의 색상을 리사이징하여 각 포인트에 매핑
image_resized_for_colors = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
colors_bgr = image_resized_for_colors.reshape(-1, 3) # (픽셀 수, 3) BGR
# Open3D는 RGB (0.0-1.0)을 기대하므로 변환
colors_rgb_normalized = colors_bgr[:, ::-1].astype(np.float64) / 255.0 # BGR -> RGB, 0-1 정규화

pcd.colors = o3d.utility.Vector3dVector(colors_rgb_normalized)

# 3. 뷰어 열기
print("\nOpen3D 뷰어 창이 열립니다. 마우스를 드래그하여 3D 포인트를 회전할 수 있습니다.")
o3d.visualization.draw_geometries([pcd])
# -------------------------------------------------------------------------

# 결과출력 (기존 DepthMap 시각화)
# cv2.imshow('Original Image', image)
# cv2.imshow('DepthMap', cv2.applyColorMap(gray, cv2.COLORMAP_JET))

cv2.waitKey(0)
cv2.destroyAllWindows()