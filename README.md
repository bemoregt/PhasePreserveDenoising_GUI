# Phase Preserve Denoising GUI

위상 보존 디노이징(Phase Preserve Denoising) 알고리즘을 구현한 GUI 애플리케이션입니다. 이 애플리케이션은 이미지 노이즈 제거 방법을 시각적으로 비교할 수 있습니다.

## 결과 예시

### 예시 1
![PhasePreserveDenoising 예시 1](https://github.com/bemoregt/PhasePreserveDenoising_GUI/blob/main/aaa.png)

### 예시 2
![PhasePreserveDenoising 예시 2](https://github.com/bemoregt/PhasePreserveDenoising_GUI/blob/main/bbb.png)

## 주요 기능

- 이미지 로드 및 가우시안 노이즈 적용
- 양방향(Bilateral) 필터링을 이용한 노이즈 제거
- 위상 보존 디노이징(Phase Preserve Denoising) 알고리즘을 이용한 노이즈 제거
- 세 가지 이미지(노이즈 이미지, Bilateral 필터링, 위상 보존 디노이징) 시각적 비교

## 알고리즘 설명

### 위상 보존 디노이징(Phase Preserve Denoising)

위상 보존 디노이징 알고리즘은 이미지의 주파수 영역에서 작동하며, 이미지의 위상 정보를 보존하면서 노이즈를 제거합니다. 이 알고리즘은 다음과 같은 과정으로 동작합니다:

1. 이미지를 주파수 영역으로 변환 (FFT)
2. 여러 방향과 스케일에서 Log-Gabor 필터를 적용
3. 각 주파수 대역에서 임계값을 초과하는 신호를 보존하고 노이즈 제거
4. 최종 결과를 실수 영역으로 변환하여 디노이즈된 이미지 생성

이 방법은 엣지와 텍스처 같은 중요한 이미지 특징(위상 정보)을 보존하면서 노이즈를 효과적으로 제거할 수 있습니다.

### 양방향(Bilateral) 필터링

양방향 필터링은 이미지의 공간 영역에서 작동하며, 픽셀의 위치와 값 모두에 기반하여 가중치를 적용합니다. 픽셀 값이 비슷한 이웃 픽셀들에게 더 높은 가중치를 주어 에지를 보존하면서 노이즈를 제거합니다.

## 구현 세부 사항

### ppdenoise 함수 파라미터:

- `k`: 임계값 계수. 높을수록 더 많은 노이즈 제거, 더 적은 신호 보존 (기본값: 2)
- `nscale`: 사용할 스케일 수 (기본값: 5)
- `mult`: 스케일 간 간격 계수 (기본값: 2.5)
- `norient`: 사용할 방향 수 (기본값: 6)
- `softness`: 에지 보존 강도. 낮을수록 더 강한 에지 보존 (기본값: 1.0)
- `brightness_factor`: 결과 이미지 밝기 조정 (기본값: 1.0)

### GUI 기능:

- 이미지 파일 로드
- 노이즈 수준 선택 (낮음, 중간, 높음, 매우 높음)
- 노이즈 이미지, Bilateral 필터링, 위상 보존 디노이징 결과 실시간 비교

## 요구사항

- Python 3.6+
- numpy
- matplotlib
- OpenCV (cv2)
- tkinter
- PIL (Pillow)

## 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/bemoregt/PhasePreserveDenoising_GUI.git
cd PhasePreserveDenoising_GUI

# 필요한 패키지 설치
pip install numpy matplotlib opencv-python pillow

# 애플리케이션 실행
python phase_preserve_denoising.py
```

## 사용 방법

1. 애플리케이션을 실행합니다.
2. "이미지 로드" 버튼을 클릭하여 그레이스케일 이미지를 선택합니다.
3. 노이즈 강도 드롭다운 메뉴에서 원하는 노이즈 수준을 선택합니다.
4. 세 가지 결과 이미지(노이즈 이미지, Bilateral 필터링, 위상 보존 디노이징)를 비교합니다.

## 참고

- 위상 보존 디노이징 알고리즘은 계산 비용이 높을 수 있으며, 큰 이미지에서는 처리 시간이 길어질 수 있습니다.
- 현재 구현은 그레이스케일 이미지만 지원합니다.

## 라이선스

MIT License

## 개발자

- [bemoregt](https://github.com/bemoregt)
