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


 ####심화코드: DepthMap을기반으로3D 포인트클라우드생성
 #실제 3D 스캐닝 데이터는 아니지만, 
 # 2D 이미지의 밝기 정보를 3D 공간의 깊이 정보로 '가정'하여
 # 점들의 집합(포인트 클라우드)을 만드는 개념을 보여줍니다
 
 
import cv2
import numpy as np

# 이미지 로드
image= cv2.imread('sample2.jpg')

# 그레이스케일 변환
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# DepthMapb생성
depth_map= cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# 3D 포인트 클라우드 변환
h, w= depth_map.shape[:2]
X, Y= np.meshgrid(np.arange(w), np.arange(h))
# np.arange(w): 0부터 w-1까지의 정수 배열
# np.arange(h): 0부터 h-1까지의 정수 배열
# np.meshgrid: 2D 그리드 좌표 생성
Z= gray.astype(np.float32)  # Depth값(이미지의 밝기값)을 Z축으로 사용
# 밝기가 클수록 가까이 있는 것으로 가정

# 3D 좌표생성
points_3d = np.dstack((X, Y, Z))
# 배열을 Z축으로 쌓아 3D 좌표 생성


# 결과출력
cv2.imshow('DepthMap', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()



