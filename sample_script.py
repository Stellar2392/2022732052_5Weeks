import numpy as np
import matplotlib.pyplot as plt
import cv2

# 그래프 폰트 및 마이너스 기호 깨짐 방지 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 데이터 생성 및 통계량 계산
# ==========================================
np.random.seed(0) # 매번 동일한 난수가 나오도록 시드 고정
# 평균(loc) 50, 표준편차(scale) 10인 정규분포를 따르는 난수 1000개 생성
data = np.random.normal(loc=50, scale=10, size=1000)

mean_value = np.mean(data) # 데이터의 평균 계산
std_value = np.std(data)   # 데이터의 표준편차 계산

print(f"Mean of data: {mean_value:.2f}")
print(f"Standard deviation of data: {std_value:.2f}")

# ==========================================
# 2. 데이터 시각화 (Matplotlib)
# ==========================================
plt.figure(figsize=(10, 5)) # 전체 그래프 창의 크기 설정

# 좌측 그래프: 히스토그램 (전체 데이터의 분포를 확인)
plt.subplot(1, 2, 1) # 1행 2열의 첫 번째 영역 지정
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
# 평균값을 나타내는 수직선(axvline) 추가
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.title("Data Distribution (Histogram)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

# 우측 그래프: 선 그래프 (데이터의 개별 값 변화를 확인)
plt.subplot(1, 2, 2) # 1행 2열의 두 번째 영역 지정
# 전체 데이터 중 처음 100개만 슬라이싱하여 시각화
plt.plot(data[:100], marker='o', linestyle='-', color='purple', alpha=0.7)
# 평균값을 나타내는 수평선(axhline) 추가
plt.axhline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.title("Sample Data (Line Graph)")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()

plt.tight_layout() # 그래프 간의 간격을 보기 좋게 자동 조정
plt.show()

# ==========================================
# 3. 이미지 로드 및 출력 (OpenCV + Matplotlib)
# ==========================================
image_path = "/mnt/data/image.png"
image = cv2.imread(image_path) # OpenCV를 사용해 이미지 읽기 (기본적으로 BGR 색상 배열로 불러옴)

if image is not None: # 이미지가 정상적으로 불러와졌는지 확인
    # OpenCV의 BGR 포맷을 Matplotlib에서 정상적인 색상으로 출력하기 위해 RGB 포맷으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image) # 이미지 출력
    plt.axis('off')   # x축, y축 눈금선 숨기기
    plt.title("Loaded Image")
    plt.show()
else:
    # 지정한 경로에 파일이 없을 경우의 예외 처리
    print("Failed to load the image.")
