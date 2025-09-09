import numpy as np
from sklearn.mixture import GaussianMixture
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def 픽셀선택(mean): #사진에 표기를 위한 함수
    max_density_x = mean[0]
    max_density_y = mean[1]
    start_x = int(max_density_x - 60)
    end_x = int(max_density_x + 60)
    start_y = int(max_density_y - 60)
    end_y = int(max_density_y + 60)
    return start_x, end_x, start_y, end_y

def 색칠(img_rgb, start_x, end_x, start_y, end_y): #사진에 표기를 위한 함수
    img_rgb[start_y:end_y, start_x:end_x] = [255, 0, 0]
    return img_rgb

def 사진저장(img_rgb): #배열 -> 사진으로 저장 (사진출력이란 이름은 배열을 사진으로 보여주는 것으로 오해할 수 있음)
    cv2.imwrite('output.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def G값시각화(mask):
    plt.imshow(mask, cmap='gray')
    plt.colorbar(label='mask value')
    plt.show()

def 가우시안분포생성(mean, cov, mask): #가우시안 분포: 2차원 정규분포(밀도를 나타냄)
    # 격자 생성
    x = np.linspace(0, mask.shape[1], 100)
    y = np.linspace(0, mask.shape[0], 100)
    X, Y = np.meshgrid(x, y)

    # 2D 가우시안 분포 생성
    rv = multivariate_normal(mean, cov)

    # 격자 상의 각 점에서의 확률 밀도 계산
    Z = rv.pdf(np.dstack((X, Y)))

    # 가우시안 분포 시각화
    plt.contour(X, Y, Z)
    plt.show()

def image_process(img):
    """이미지 처리 및 GMM 적합을 수행"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 이미지의 색상 공간을 변환
    R, G, B = cv2.split(img_rgb) # RGB 분리
    mask = np.where(((G - np.maximum(R, B)) > 50) & (G > np.maximum(R, B)), 1, 0)

    y_indices, x_indices = np.where(mask == 1) # 값이 1인것만 뽑아내기
    coords = np.column_stack((x_indices, y_indices)) # 2차원 배열로 변환

    # GMM적합 (굳이 GMM 적합을 두번 할 필요가 없음, 함수를 제거하여 중복되는 코드 삭제)
    gmm = GaussianMixture(n_components=1, covariance_type='full').fit(coords)
    # 평균/공분산 계산
    mean = gmm.means_[0] # type: ignore
    cov = gmm.covariances_[0] # type: ignore
    
    return img_rgb, mask, mean, cov

def 사진반환(img): #표기된 사진 배열 생성
    img_rgb, _, mean, _ = image_process(img)
    
    start_x, end_x, start_y, end_y = 픽셀선택(mean) # 밀도가 가장 높은 픽셀 선택
    img_rgb = 색칠(img_rgb, start_x, end_x, start_y, end_y) # 빨간색으로 표시
    return img_rgb

if __name__ == "__main__": # 위 조건문이 없으면 import 할 때도 실행될 수 있음 만약을 위해 써주기
    img = cv2.imread('image.jpg') # 이미지

    if img is not None:
        img_rgb, mask, mean, cov = image_process(img)

        start_x, end_x, start_y, end_y = 픽셀선택(mean) # 밀도가 가장 높은 픽셀 선택
        img_rgb = 색칠(img_rgb, start_x, end_x, start_y, end_y) # 빨간색으로 표시

        가우시안분포생성(mean, cov, mask)# 가우시안 분포 시각화

        # mask 값 시각화
        G값시각화(mask)

        plt.imshow(img_rgb)
        plt.show() #사진출력

        사진저장(img_rgb) # 변경된 이미지 저장
    else:
        raise Exception("입력한 이미지를 찾을 수 없습니디.")