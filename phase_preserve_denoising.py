import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# ppdenoise 함수 정의 (기존 코드 그대로 사용)
def ppdenoise(
        img,
        k=2,  # 2에서 1.5로 줄임 - 더 적은 노이즈 제거로 더 많은 신호 보존
        nscale=5,
        mult=2.5,  # 2.5에서 2.0으로 줄임 - 스케일 간 갭 축소
        norient=6,
        softness=1.0,  # 1.0에서 0.8로 줄임 - 더 강한 에지 보존
        brightness_factor=1.0):  # 결과 이미지의 밝기를 증가시키는 새 파라미터
    """Function to denoise an image while preserving phase information"""
    min_wavelength = 2
    sigma_onf = 0.55
    dtheta_onsigma = 1.0
    epsilon = 0.00001
    theta_sigma = np.pi/norient/dtheta_onsigma
    img_fft = np.fft.fft2(img)
    row, col = img_fft.shape
    x = np.matmul(
            np.ones((row, 1)),
            (np.arange(
                -col/2, col/2)/(col/2)).reshape(1, -1))
    y = np.matmul(
            np.arange(-row/2, row/2).reshape(-1, 1),
            np.ones((1, col))/(row/2)
            )
    radius = np.sqrt(np.square(x)+np.square(y))
    radius[int(row/2), int(col/2)] = 1
    theta = np.arctan2(-y, x)
    total_energy = np.zeros((row, col))
    estmean_en = []
    sig = []
    for orient in np.arange(1, norient+1):
        print("Processing orientation {}".format(orient))
        angl = (orient - 1)*np.pi/norient
        wavelength = min_wavelength
        ds = np.subtract(
                np.sin(theta)*np.cos(angl),
                np.cos(theta)*np.sin(angl))
        dc = np.add(
                np.cos(theta)*np.cos(angl),
                np.sin(theta)*np.sin(angl))
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-np.square(dtheta)/(2*theta_sigma**2))
        for scale in np.arange(1, nscale+1):
            f_o = 1.0/wavelength
            rf_o = f_o/0.5
            log_gabor = np.exp(
                    -np.square(
                        np.log(
                            radius/rf_o))/(2*np.log(sigma_onf)**2)
                        )
            log_gabor[int(row/2), int(col/2)] = 0
            filter = log_gabor*spread
            filter = np.fft.fftshift(filter)
            e0_fft = img_fft*filter
            e0 = np.fft.ifft2(e0_fft)
            ae0 = np.abs(e0)
            if scale == 1:
                median_en = np.median(ae0.reshape(1, row*col))
                mean_en = (0.5*np.sqrt(-np.pi/np.log(0.5))) * median_en
                ray_var = (4-np.pi)*np.square(mean_en)/np.pi
                ray_mean = mean_en
                estmean_en.append(mean_en)
                sig.append(np.sqrt(ray_var))
            t = (ray_mean + k*np.sqrt(ray_var))/(np.power(mult, scale-1))
            valid_e0 = (ae0 > t)
            v = np.divide(softness*t*e0, (ae0+epsilon))
            v = np.add(
                    np.multiply(
                        np.invert(valid_e0).astype(int),
                        e0),
                    np.multiply(
                        valid_e0.astype(int),
                        v)
                    )
            e0 = e0-v
            total_energy = total_energy + e0
            wavelength = wavelength*mult
    
    # 실수 부분만 추출하고 밝기 조정
    clean_image = np.real(total_energy) * brightness_factor
    
    # 값 범위를 0-1로 클리핑
    clean_image = np.clip(clean_image, 0, 1)
    
    return clean_image

class ImageDenoiseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 노이즈 제거 비교")
        self.root.geometry("1200x700")
        
        # 한글 폰트 설정
        self.setup_font()
        
        # matplotlib 전역 폰트 설정
        plt.rcParams['font.family'] = self.font_family
        
        # 이미지 변수 초기화
        self.original_img = None
        self.noisy_img = None
        self.bilateral_img = None
        self.ppdenoise_img = None
        
        # 프레임 생성
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(pady=10)
        
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 기본 이미지 로드 (샘플 이미지)
        self.default_img = self.create_sample_image()
        
        # 컨트롤 버튼들
        self.load_btn = tk.Button(self.top_frame, text="이미지 로드", command=self.safe_load_image)
        self.load_btn.grid(row=0, column=0, padx=10)
        
        tk.Label(self.top_frame, text="노이즈 강도:", font=self.font).grid(row=0, column=1, padx=10)
        self.noise_var = tk.StringVar()
        self.noise_levels = ["낮음", "중간", "높음", "매우 높음"]
        self.noise_combo = ttk.Combobox(self.top_frame, textvariable=self.noise_var, values=self.noise_levels, font=self.font)
        self.noise_combo.current(0)
        self.noise_combo.grid(row=0, column=2, padx=10)
        self.noise_combo.bind("<<ComboboxSelected>>", self.update_noise)
        
        # 그림 영역 생성 (matplotlib 사용)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 초기 그림 설정
        for ax in self.axes:
            ax.axis('off')
        self.axes[0].set_title("노이즈 이미지")
        self.axes[1].set_title("Bilateral 필터링")
        self.axes[2].set_title("위상 보존 디노이징")
        self.fig.tight_layout()
        
        # 초기 이미지로 시작
        self.original_img = self.default_img.copy()
        self.update_noise(None)
    
    def setup_font(self):
        # 폰트 설정
        if sys.platform.startswith('darwin'):  # macOS
            self.font = ('AppleGothic', 12)
            self.font_family = 'AppleGothic'
        elif sys.platform.startswith('win'):  # Windows
            self.font = ('Malgun Gothic', 12)
            self.font_family = 'Malgun Gothic'
        else:  # Linux 등 기타 OS
            self.font = ('TkDefaultFont', 12)
            self.font_family = 'DejaVu Sans'
    
    def create_sample_image(self):
        # 샘플 그레이스케일 이미지 생성 (체스보드 패턴)
        img = np.zeros((256, 256), dtype=np.float32)
        block_size = 32
        for i in range(0, 256, block_size):
            for j in range(0, 256, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    img[i:i+block_size, j:j+block_size] = 0.9
                else:
                    img[i:i+block_size, j:j+block_size] = 0.3
                    
        # 가장자리에 원 추가
        center = (128, 128)
        radius = 100
        y, x = np.ogrid[:256, :256]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = img[mask] * 0.7 + 0.15
        
        return img
    
    def safe_load_image(self):
        try:
            # 이미지 파일 선택 대화상자 - MacOS에서 문제가 있으므로 간단하게 처리
            file_path = filedialog.askopenfilename()
            
            if not file_path:
                return
            
            # 이미지 로드 및 그레이스케일 변환
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("이미지를 불러올 수 없습니다.")
                
            self.original_img = img.astype(np.float32) / 255.0  # 0-1 정규화
            
            # 노이즈 수준으로 업데이트
            self.update_noise(None)
            
        except Exception as e:
            # 오류 발생 시 기본 이미지 사용
            print(f"이미지 로드 오류: {e}")
            self.original_img = self.default_img.copy()
            self.update_noise(None)
    
    def add_noise(self, img, level):
        # 노이즈 수준에 따라 가우시안 노이즈 추가
        noise_levels = {
            "낮음": 0.1,
            "중간": 0.5,
            "높음": 0.9,
            "매우 높음": 1.5
        }
        sigma = noise_levels.get(level, 0.01)
        
        # 가우시안 노이즈 생성
        noisy_img = img.copy()
        noise = np.random.normal(0, sigma, img.shape)
        noisy_img = noisy_img + noise
        
        # 값 범위 제한 (0-1)
        noisy_img = np.clip(noisy_img, 0, 1)
        return noisy_img
    
    def update_noise(self, event):
        if self.original_img is None:
            return
            
        noise_level = self.noise_var.get()
        if not noise_level:  # 콤보박스가 비어있을 경우 기본값 설정
            noise_level = "낮음"
            self.noise_var.set(noise_level)
        
        # 노이즈 이미지 생성
        self.noisy_img = self.add_noise(self.original_img, noise_level)
        
        # Bilateral 필터링 적용
        # 먼저 0-255 범위로 변환
        temp_img = (self.noisy_img * 255).astype(np.uint8)
        bilateral_result = cv2.bilateralFilter(temp_img, 9, 75, 75)
        self.bilateral_img = bilateral_result.astype(np.float32) / 255.0
        
        # 위상 보존 디노이징 적용 - 계산이 길어질 수 있으므로 try-except로 보호
        try:
            self.ppdenoise_img = ppdenoise(self.noisy_img)
            # 값 범위 제한 (0-1)
            self.ppdenoise_img = np.clip(self.ppdenoise_img, 0, 1)
        except Exception as e:
            print(f"위상 보존 디노이징 오류: {e}")
            self.ppdenoise_img = self.noisy_img.copy()
        
        # 이미지 업데이트
        self.update_images()
    
    def update_images(self):
        # 모든 축 지우기
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
        
        # 이미지 표시
        self.axes[0].imshow(self.noisy_img, cmap='gray')
        self.axes[0].set_title("노이즈 이미지")
        
        self.axes[1].imshow(self.bilateral_img, cmap='gray')
        self.axes[1].set_title("Bilateral 필터링")
        
        self.axes[2].imshow(self.ppdenoise_img, cmap='gray')
        self.axes[2].set_title("위상 보존 디노이징")
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDenoiseApp(root)
    root.mainloop()
