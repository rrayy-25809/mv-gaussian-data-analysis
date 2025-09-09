import numpy as np
from sklearn.mixture import GaussianMixture
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def 가우시안적합(coords):
    gmm = GaussianMixture(n_components=1, covariance_type='full').fit(coords)
    mean = gmm.means_[0]
    return mean

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

def 사진출력(img_rgb): #배열 -> 사진 변환
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

def 사진반환(img): #표기된 사진 배열 생성
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(img_rgb)
    mask = np.where(((G - np.maximum(R, B)) > 50) & (G > np.maximum(R, B)), 1, 0)
    y_indices, x_indices = np.where(mask == 1)
    coords = np.column_stack((x_indices, y_indices))
    mean = 가우시안적합(coords)
    start_x, end_x, start_y, end_y = 픽셀선택(mean)
    img_rgb = 색칠(img_rgb, start_x, end_x, start_y, end_y)
    return img_rgb

# 이미지
img = cv2.imread('image.jpg')

# RGB값 저장
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# R, G, B 분리
R, G, B = cv2.split(img_rgb)
# G에서 R과 B 중 더 큰 값을 뺀 값이 50 이상이고, RGB중 G가 가장 큰것은 1로, 아니면 0으로 배열생성
mask = np.where(((G - np.maximum(R, B)) > 50) & (G > np.maximum(R, B)), 1, 0)

# 값이 1인것만 뽑아내기
y_indices, x_indices = np.where(mask == 1)

# 2차원 배열로 변환
coords = np.column_stack((x_indices, y_indices))

# GMM 적합 및 밀도가 가장 큰 픽셀의 좌표 계산
mean = 가우시안적합(coords)

# 밀도가 가장 높은 픽셀 선택
start_x, end_x, start_y, end_y = 픽셀선택(mean)

# 빨간색으로 표시
img_rgb = 색칠(img_rgb, start_x, end_x, start_y, end_y)

# 변경된 이미지 저장
사진출력(img_rgb)



# GMM적합
gmm = GaussianMixture(n_components=1, covariance_type='full').fit(coords)
# 평균/공분산 계산
mean = gmm.means_[0]
cov = gmm.covariances_[0]
# 가우시안 분포 시각화
가우시안분포생성(mean, cov, mask)

# mask 값 시각화
G값시각화(mask)

#사진출력
plt.imshow(img_rgb)
plt.show()