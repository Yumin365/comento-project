 #pytest를 활용한 기본 UnitTest
import numpy as np
import pytest
import cv2
 # 샘플함수: 가짜깊이맵생성
def generate_depth_map(image):
    if image is None: #이미지가 제대로 로드되지 않았거나 전달되지 않은 경우
        raise ValueError("입력된 이미지가 없습니다.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 이미지를 grayscale로 변환

    # 가짜 깊이 맵 적용
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET) #grayscale에 특정 컬러맵 적용시킴
    return depth_map


# 테스트 코드
def test_generate_depth_map():
    image= np.zeros((100, 100, 3), dtype=np.uint8)  #테스트용 가상 검정색 빈 이미지 (100x100, 3채널, 8비트 정수)
    depth_map= generate_depth_map(image)
    # 1. 입력 이미지가 없을 경우 예외 처리 확인 테스트
    # depth_map= generate_depth_map(None)

    
    assert depth_map.shape== image.shape, "출력 크기가 입력 크기와 다릅니다." # assert: 조건이 True가 아니면 AssertionError 발생
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."
 
 
 
 # pytest실행
if __name__ == "__main__":
    pytest.main() # test_로 시작하는 함수들을 자동으로 찾아서 실행