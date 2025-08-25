import numpy as np
import sounddevice as sd
from ai_edge_litert.interpreter import Interpreter
import scipy.signal
import collections
import math
import csv
import zipfile
import RPi.GPIO as GPIO
import time
from PIL import Image, ImageDraw, ImageFont
import ST7735

VIBRATION_PIN = 13
# ---- 하드웨어 세팅 ----
CHUNK_DURATION = 2.0
SLIDE_STEP = 1.0
ORIGINAL_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
MIC_DISTANCE = 0.15  # 3cm 마이크 거리 (실측값 필요)
SPEED_OF_SOUND = 343.0
MIN_ENERGY_THRESHOLD = 0.01
MODEL_INPUT_SAMPLES = 15600
SMOOTHING_WINDOW = 2
NORMALIZATION_BIT_DEPTH = 32
MAX_VAL_FOR_NORMALIZATION = 2**(NORMALIZATION_BIT_DEPTH - 1)
MIN_RMS_THRESHOLD = 0.02  # 반응할 최소 소리 크기 (소음 필터링)
DIRECTION_LABELS = ["왼쪽", "오른쪽"] # 채널 0, 1에 해당하는 방향
MODEL_INPUT_SAMPLES = 15600

disp = ST7735.ST7735(
    port=0,           # SPI 포트 (일반적으로 0)
    cs=0,             # SPI CS (일반적으로 0)
    dc=25,            # Data/Command 핀 (예: GPIO 24)
    backlight=24, # 백라이트 핀 (없으면 None)
    rst=27,           # Reset 핀 (예: GPIO 25)
    width=128,        # 디스플레이 가로 해상도
    height=128,       # 디스플레이 세로 해상도
    rotation=0,
    offset_left = 2,
    offset_top = 1,# 회전 각도 (필요에 따라 0, 90, 180, 270)
    invert=False      # 색상 반전 여부
)

WIDTH = disp.width
HEIGHT = disp.height

sound_category = {
    # 1. 비상 경보 및 사이렌
    317: '1',  # 경찰차(사이렌)
    318: '1',  # 구급차(사이렌)
    319: '1',  # 소방차(사이렌)
    390: '1',  # 사이렌
    391: '1',  # 민방위 사이렌
    392: '1',  # 부저(삑삑이)
    393: '1',  # 연기 감지기/화재경보음
    394: '1',  # 화재 경보
    396: '1',  # 휘슬

    # 2. 교통·차량 관련 위험
    302: '2',  # 차량 경적, 자동차 경적
    303: '2',  # 빵빵(경적)
    304: '2',  # 자동차 알람
    312: '2',  # 에어 혼, 트럭 경적
    325: '2',  # 기차 경적
    306: '2',  # 미끄러짐(타이어)
    307: '2',  # 타이어 끽끽거림

    # 3. 폭발·총성·화재·자연재해
    420: '3',  # 폭발
    421: '3',  # 총성
    422: '3',  # 기관총 발사음
    423: '3',  # 연속 사격
    424: '3',  # 대포 발사음
    428: '3',  # 터지는 소리, 펑 소리
    429: '3',  # 분출, 폭발
    430: '3',  # 쾅(폭발음)

    # 4. 사고·파손·충돌
    433: '4',  # 찍는 소리
    434: '4',  # 금 가는 소리
    437: '4',  # 산산조각, 깨짐
    454: '4',  # 쿵, 툭 소리
    455: '4',  # 툭 소리
    460: '4',  # 쾅 소리
    462: '4',  # 퍽, 휙 소리
    463: '4',  # 박살, 충돌
    464: '4',  # 깨짐
    478: '4',  # 쨍그랑 소리
    480: '4',  # 삐걱거림
    483: '4',  # 덜그럭거림
    486: '4',  # 덜컹덜컹 소리

    # 5. 경고·주의 신호
    313: '5',  # 후진 경고음
    475: '5',  # 삑삑(경고음)
    355: '5',  # 끽끽거림

    # 6. 사람의 위험 신호
    6:   '6',  # 외침
    7:   '6',  # 고함, 울부짖음
    9:   '6',  # 소리침
    10:  '6',  # 아이들 소리침
    11:  '6',  # 비명
    479: '6',  # 끽끽, 비명 소리
}

duty_dict = {
    '1': [20, 50, 30, 0],
    '2': [20,20,20,20],
    '3': [50,0,50,0],
    '4': [10,20,30,40,50],
    '5': [50,40,30,20,10],
    '6': [50,50,50,50]
    }

direction = 0
# 이미지 객체 생성
img = Image.new('RGB', (WIDTH, HEIGHT), (0,0,0))
draw = ImageDraw.Draw(img)

# 기본 폰트 로드
font = ImageFont.load_default()


def gcc_phat(sig1, sig2, sr):
    n = sig1.shape[0] + sig2.shape[0]
    fft_sig1 = np.fft.rfft(sig1, n=n)
    fft_sig2 = np.fft.rfft(sig2, n=n)
    R = fft_sig1 * np.conj(fft_sig2)
    cc = np.fft.irfft(R / (np.abs(R) + 1e-10))
    max_shift = int(sr * MIC_DISTANCE / SPEED_OF_SOUND)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    lag_idx = np.argmax(np.abs(cc)) - max_shift
    delay = lag_idx / sr
    return delay

def calculate_angle_tdoa(delay):
    x = SPEED_OF_SOUND * delay
    angle_rad = math.asin(max(-1, min(1, x / MIC_DISTANCE)))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def calculate_rms(audio_chunk):
    """오디오 청크의 RMS(에너지)를 계산합니다."""
    return np.sqrt(np.mean(audio_chunk**2))

def get_direction_from_sign_tdoa(left_channel, right_channel):
    """
    Sign-of-TDOA를 계산하여 방향을 문자열로 반환합니다.
    - 왼쪽에서 소리가 먼저 도달하면 "왼쪽"
    - 오른쪽에서 소리가 먼저 도달하면 "오른쪽"
    - 거의 동시에 도달하면 "불명확"
    """
    # 1. 두 채널의 상호 상관(Cross-correlation) 계산
    # 이 연산이 시간차를 찾는 핵심입니다.
    corr = np.correlate(left_channel, right_channel, mode='full')

    # 2. 상관관계가 가장 높은 지점의 인덱스 찾기
    delay_index = np.argmax(corr)

    # 3. 상관관계 배열의 중앙 인덱스 계산 (시간 지연 '0' 지점)
    center_index = (len(corr) - 1) / 2

    # 4. 시간차의 부호(Sign) 판단
    # argmax가 중앙보다 크면 오른쪽, 작으면 왼쪽에서 소리가 먼저 도달한 것입니다.
    if delay_index > center_index:
        return "오른쪽"
    elif delay_index < center_index:
        return "왼쪽"
    else:
        return "중앙/불명확"

model_path = '/home/pi/capstone/yamnet-tflite-classification-tflite-v1/1.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
result_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
labels_file = zipfile.ZipFile(model_path).open('yamnet_label_list.txt')
class_names = [l.decode('utf-8').strip() for l in labels_file.readlines()]

GPIO.setmode(GPIO.BCM)
GPIO.setup(VIBRATION_PIN, GPIO.OUT) 

# 100Hz PWM 객체 생성
pwm = GPIO.PWM(VIBRATION_PIN, 100)
pwm.start(0)  # 시작 시 진동 OFF
print(sd.query_devices())



try:
    print("2채널 실시간 소리분류+방향 탐지 시작 (Ctrl+C 종료)")
    buffer = np.zeros((int(CHUNK_DURATION * ORIGINAL_RATE), CHANNELS), dtype=np.int32)
    while True:
        new_samples = int(SLIDE_STEP * ORIGINAL_RATE)
        recording = sd.rec(new_samples, samplerate=ORIGINAL_RATE, channels=CHANNELS, dtype='int32', device='hw:3,0')
        sd.wait()
        
        
        buffer = np.roll(buffer, -new_samples, axis=0)
        buffer[-new_samples:, :] = recording

        waveform_float_multi = buffer.astype(np.float32) / MAX_VAL_FOR_NORMALIZATION
        ch0, ch1 = waveform_float_multi[:, 0], waveform_float_multi[:, 1]
        
        # 각 채널별 RMS 계산
        rms_values = [calculate_rms(waveform_float_multi[:, i]) for i in range(CHANNELS)]

        # --- 방향 탐지 (TDOA) ---
        #delay = gcc_phat(ch0, ch1, ORIGINAL_RATE)
        #angle = calculate_angle_tdoa(delay)
        
                
        # direction 판단: 마이크 0 → 왼쪽, 마이크 1 → 오른쪽
        #direction = 0 if delay < 0 else 1
        #direction_text = DIRECTION_LABELS[direction]
        direction = get_direction_from_sign_tdoa(ch0, ch1)
        
        # --- 분류 입력 준비 (모노) ---
        waveform_mono = np.mean(waveform_float_multi, axis=1)
        
        peak_energy = np.max(np.abs(waveform_mono))
        
        if peak_energy < MIN_ENERGY_THRESHOLD:
            print(f"소리 감지 대기 중... (에너지: {peak_energy:.4f})", end='\r')
            continue
        waveform_scaled = waveform_mono / peak_energy
        
        if ORIGINAL_RATE != TARGET_RATE:
            num_target = int(len(waveform_scaled) * TARGET_RATE / ORIGINAL_RATE)
            waveform_resampled = scipy.signal.resample(waveform_scaled, num_target)
        else:
            waveform_resampled = waveform_scaled
        if len(waveform_resampled) > MODEL_INPUT_SAMPLES:
            waveform_resampled = waveform_resampled[:MODEL_INPUT_SAMPLES]
        elif len(waveform_resampled) < MODEL_INPUT_SAMPLES:
            padding = np.zeros(MODEL_INPUT_SAMPLES - len(waveform_resampled))
            waveform_resampled = np.concatenate([waveform_resampled, padding])
        waveform_resampled = np.squeeze(waveform_resampled).astype(np.float32)
        


        interpreter.set_tensor(input_details[0]['index'], waveform_resampled)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        
        input_shape = input_details[0]['shape']  # 예: [15600] 또는 [1, 15600]
        if len(input_shape) == 2:
            input_data = np.expand_dims(waveform_resampled, axis=0).astype(np.float32)
        else:
            input_data = waveform_resampled.astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
            
        top_class_index = np.argmax(scores)
        predicted_class = class_names[top_class_index]
        confidence = scores[top_class_index]
        result_buffer.append(predicted_class)
        most_common = max(set(result_buffer), key=result_buffer.count)

        #print(f"감지:{most_common:<20} | 방향:{direction_text} (신뢰도:{confidence:.2f})", end='\r')
        # 각 채널 신호/angle/지연 실시간 모니터링
        #print(f"delay: {delay:.6f}s, angle: {angle:.2f}, ch0_rms: {rms_values[0]:.3f}, ch1_rms: {rms_values[1]:.3f}")
        print(f"direction: {direction}, ch0_rms: {rms_values[0]:.3f}, ch1_rms: {rms_values[1]:.3f}")
        
        

except KeyboardInterrupt:
    print("\n프로그램 종료.")
except Exception as e:
    print(f"\n오류 발생: {e}")
    
    '''
#텍스트 크기 측정
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

#중앙 좌표 계산
x = (WIDTH - text_width) // 2
y = (HEIGHT - text_height) // 2

#텍스트 그리기
draw.text((x, y), text, font=font, fill=(255, 255, 255))'''
