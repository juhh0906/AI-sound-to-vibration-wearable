from ai_edge_litert.interpreter import Interpreter
import sys
import numpy as np
import sounddevice as sd
import scipy.signal
import csv
import collections
import zipfile
import math
import RPi.GPIO as GPIO
import time
from PIL import Image, ImageDraw, ImageFont
import ST7735

print("asdf")

VIBRATION_PIN = 13
# --- 설정값 ---
CHUNK_DURATION = 2.0
SLIDE_STEP = 1.0
ORIGINAL_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
# MIN_ENERGY_THRESHOLD = 0.01
NORMALIZATION_BIT_DEPTH = 32
MAX_VAL_FOR_NORMALIZATION = 2 ** (NORMALIZATION_BIT_DEPTH - 1)
MIN_RMS_THRESHOLD = 0.01  # 반응할 최소 소리 크기 (소음 필터링)
DIRECTION_LABELS = ["왼쪽", "오른쪽"]  # 채널 0, 1, 2에 해당하는 방향
# TFLITE_MODEL_PATH = "yamnet_quant.tflite"
MODEL_INPUT_SAMPLES = 15600
SMOOTHING_WINDOW = 2
MIC_DISTANCE_CM = 30.0        # 마이크 사이의 거리 (cm) - **실제 값으로 수정 필수**
SPEED_OF_SOUND_CM_S = 34300   # 소리의 속도 (cm/s)
MIC_DISTANCE_M = MIC_DISTANCE_CM / 100.0
MAX_TDOA = MIC_DISTANCE_M / (SPEED_OF_SOUND_CM_S / 100.0)
STOP_BUTTON_PIN = 17
START_BUTTON_PIN = 22

disp = ST7735.ST7735(
    port=0,           # SPI 포트 (일반적으로 0)
    cs=0,             # SPI CS (일반적으로 0)
    dc=25,            # Data/Command 핀 (예: GPIO 24)
    backlight=24,     # 백라이트 핀 (없으면 None)
    rst=27,           # Reset 핀 (예: GPIO 25)
    width=128,        # 디스플레이 가로 해상도
    height=128,       # 디스플레이 세로 해상도
    rotation=180,
    offset_left=2,
    offset_top=1,     # 회전 각도 (필요에 따라 0, 90, 180, 270)
    invert=False      # 색상 반전 여부
)

category_image = {
    '1': '/home/pi/capstone/1.jpg',
    '2': '/home/pi/capstone/2.jpg',
    '3': '/home/pi/capstone/3.jpg',
    '4': '/home/pi/capstone/4.jpg',
    '5': '/home/pi/capstone/5.jpg',
    '6': '/home/pi/capstone/6.jpg',
}

WIDTH = disp.width
HEIGHT = disp.height
scaled_width =32
scaled_height =32
# 이미지 객체 생성
img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

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
    6: '6',    # 외침
    7: '6',    # 고함, 울부짖음
    9: '6',    # 소리침
    10: '6',   # 아이들 소리침
    11: '6',   # 비명
    479: '6',  # 끽끽, 비명 소리
}

duty_dict = {
    '1': [20, 50, 30, 0],
    '2': [20, 20, 20, 20],
    '3': [50, 0, 50, 0],
    '4': [10, 20, 30, 40, 50],
    '5': [50, 40, 30, 20, 10],
    '6': [50, 50, 50, 50]
}

def display_startup_image(image_path, duration=3):
    """시작화면 이미지를 디스플레이에 표시하는 함수"""
    try:
        # 이미지 파일 로드
        startup_img = Image.open(image_path)

        # 디스플레이 크기에 맞게 리사이즈 (128x128)
        startup_img = startup_img.resize((WIDTH, HEIGHT), Image.LANCZOS)

        # 디스플레이에 표시
        disp.display(startup_img)

        # 지정된 시간만큼 대기
        time.sleep(duration)

    except Exception as e:
        print(f"시작화면 이미지 로드 실패: {e}")
        
        # 이미지 로드 실패 시 기본 display_message("TEAM S.O.S.")
        time.sleep(duration)


def display_message(message):
    """디스플레이 중앙에 메시지를 표시하는 헬퍼 함수"""
    draw.rectangle((0, 0, disp.width, disp.height), outline=0, fill=(0,0,0))
    bbox = draw.textbbox((0,0), message, font=font)
    x = (disp.width - (bbox[2] - bbox[0])) // 2
    y = (disp.height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), message, font=font, fill=(255, 255, 255))
    disp.display(img)

def calculate_rms(audio_chunk):
    """오디오 청크의 RMS(에너지)를 계산합니다."""
    return np.sqrt(np.mean(audio_chunk ** 2))

def calculate_peak_rms(audio_chunk, percentile=95):
    """오디오 청크의 RMS(에너지)를 계산합니다."""
    abs_chunk = np.abs(audio_chunk)
    threshold = np.percentile(abs_chunk, percentile)
    peaks = abs_chunk[abs_chunk >= threshold]
    if peaks.size > 0:
        # return np.sqrt(np.mean(peaks**2))
        return np.mean(peaks)
    else:
        return 0

def main_application():
    """메인 애플리케이션 로직 (실시간 소리 분석 + 진동/디스플레이 + 종료 버튼 감지)"""
    # GPIO 설정
    print("메인 애플리케이션 시작. 리소스를 초기화합니다.")
    # 정지 버튼 GPIO 설정
    GPIO.setup(STOP_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VIBRATION_PIN, GPIO.OUT)


    # 100Hz PWM 객체 생성
    pwm = GPIO.PWM(VIBRATION_PIN, 100)
    pwm.start(0)  # 시작 시 진동 OFF

    model_path = '/home/pi/capstone/yamnet-tflite-classification-tflite-v1/1.tflite'
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    result_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
    labels_file = zipfile.ZipFile(model_path).open('yamnet_label_list.txt')
    class_names = [l.decode('utf-8').strip() for l in labels_file.readlines()]
    print(sd.query_devices())
    print(f"\n실시간 소리 감지 시작 (종료 버튼: GPIO {STOP_BUTTON_PIN}, 또는 Ctrl+C)")
    buffer = np.zeros((int(CHUNK_DURATION * ORIGINAL_RATE), CHANNELS), dtype=np.int32)

    # --- 2. 메인 애플리케이션 루프 ---
    while True:
        # #######################################
        if GPIO.input(STOP_BUTTON_PIN) == GPIO.LOW:
            time.sleep(1)
            break

        #display_startup_image("/home/pi/capstone/ＴＥＡＭｓｏｓ.jpg", 0.01)

        new_samples = int(SLIDE_STEP * ORIGINAL_RATE)
        recording = sd.rec(new_samples, samplerate=ORIGINAL_RATE, channels=CHANNELS, dtype='int32', device='hw:3,0')
        sd.wait()

        # 버퍼 업데이트
        buffer = np.roll(buffer, -new_samples, axis=0)
        buffer[-new_samples:, :] = recording

        # --- 파이프라인 A: 방향 탐지 (소리 세기 비교) ---
        waveform_float_multi = buffer.astype(np.float32) / MAX_VAL_FOR_NORMALIZATION
        rms_values = [calculate_peak_rms(waveform_float_multi[:, i]) for i in range(CHANNELS)]
        rms_values[0] *= 1.2 # compensation
        loudest_channel_index = np.argmax(rms_values)
        max_rms = rms_values[loudest_channel_index]

        # 소리가 임계값보다 작으면 무시
        if max_rms < MIN_RMS_THRESHOLD:
            print("소리 감지 대기 중...", " " * 40, end='\r')
            continue

        detected_direction = DIRECTION_LABELS[loudest_channel_index]

        # --- 파이프라인 B: 소리 분류 (YAMNet) ---
        waveform_mono = np.mean(waveform_float_multi, axis=1)

        # 리샘플링
        if ORIGINAL_RATE != TARGET_RATE:
            num_target = int(len(waveform_mono) * TARGET_RATE / ORIGINAL_RATE)
            waveform_resampled = scipy.signal.resample(waveform_mono, num_target)
        else:
            waveform_resampled = waveform_mono

        # 입력 길이 맞춤(패딩/자르기)
        if len(waveform_resampled) > MODEL_INPUT_SAMPLES:
            waveform_resampled = waveform_resampled[:MODEL_INPUT_SAMPLES]
        elif len(waveform_resampled) < MODEL_INPUT_SAMPLES:
            padding = np.zeros(MODEL_INPUT_SAMPLES - len(waveform_resampled))
            waveform_resampled = np.concatenate([waveform_resampled, padding])

        waveform_resampled = np.squeeze(waveform_resampled)
        input_shape = tuple(input_details[0]['shape'])
        if input_shape == (15600,):
            interpreter.set_tensor(input_details[0]['index'], waveform_resampled.astype(np.float32))
        elif input_shape == (1, 15600):
            input_tensor = np.expand_dims(waveform_resampled, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
        else:
            raise ValueError(f"예상치 못한 입력 shape: {input_shape}")

        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]['index'])

        # 출력 shape이 (1, N)일 경우 1차원으로 변환
        if len(scores.shape) == 2 and scores.shape[0] == 1:
            scores = scores[0]

        # 클래스 개수 일치 확인
        if len(scores) != len(class_names):
            print("모델 출력 shape가 비정상입니다. 입력 shape와 데이터 확인 필요.")
            continue

        top_class_index = np.argmax(scores)
        predicted_class = class_names[top_class_index]
        confidence = scores[top_class_index]

        # 후처리(다수결 스무딩)
        result_buffer.append(predicted_class)
        most_common = max(set(result_buffer), key=result_buffer.count)
        print(f" 방향:{detected_direction:<10} 최종 감지 소리: {most_common:<20} (신뢰도: {confidence:.2f})", end='\r\n')
        print(f"L채널: {rms_values[0]:.5f}         R채널: {rms_values[1]:.5f}")

        idx = class_names.index(most_common)
        display_startup_image("/home/pi/capstone/bgi.jpg", 0.01)
        if idx in sound_category.keys() and confidence >= 0.5:
            logo = Image.open("/home/pi/capstone/bgi.jpg").resize((WIDTH, HEIGHT), Image.LANCZOS)
            img.paste(logo, (0, 0))

            # 해당 카테고리 구하기
            category = sound_category[idx]
            '''
            img_path = category_image.get(category)
            if img_path:
                try:
                    base_img = Image.open(img_path).resize((WIDTH, HEIGHT), Image.LANCZOS)
                    img.paste(base_img, (48,72))
                except Exception as e:
                    print(f'이미지 로드 실패: {e}')
                    # 이미지 로드 실패시 기존 방식
                    base_img = img
                    img.paste(base_img, (48,72))
            else:
                base_img = img
                img.paste(base_img, (48,72))
            '''
            text = most_common
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (WIDTH - text_width) // 2
            y = (HEIGHT - text_height) // 2
            draw.text((x, y), text, font=font, fill=(255, 255, 255))

            # 방향 표시
            if 0.9 < rms_values[0] / rms_values[1] < 1.1:
                pass
            else:
                if detected_direction == '왼쪽':
                    draw.text((0, 68), "<-", font=font, fill=(255, 255, 255))
                else:
                    draw.text((115, 68), "->", font=font, fill=(255, 255, 255))

            disp.display(img)



            for duty in duty_dict[sound_category[idx]]:
                pwm.ChangeDutyCycle(duty)
                print(f"Duty: {duty}%")
                time.sleep(0.5)
            #draw.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=(0, 0, 0))
            #disp.display(img)
        print(".", end='', flush=True)
        # #######################################


        # --- 3. 메인 앱 리소스 정리 ---
    print("\n메인 애플리케이션 리소스를 정리합니다.")
    # 이 함수 안에서 사용한 이벤트 감지만 제거합니다.
    GPIO.remove_event_detect(STOP_BUTTON_PIN)
    pwm.stop()



# =============================================
# === 프로그램의 메인 진입점 (State Machine) ===
# =============================================
if __name__ == "__main__":
    try:
        # 최초 GPIO 모드 설정
        GPIO.setmode(GPIO.BCM)
        # 시작 버튼은 프로그램 전체에서 사용하므로 여기서 설정
        GPIO.setup(START_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        display_startup_image("/home/pi/capstone/start_image.jpeg",3)


        draw.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=(0, 0, 0))
        disp.display(img)

        while True:
            # --- "대기" 상태 ---
            print("\n[대기 모드] 시작 버튼을 누르세요...")
            #display_message("Press Start")
            draw.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=(0, 0, 0))
            disp.display(img)

            while GPIO.input(START_BUTTON_PIN):
                time.sleep(0.01)

            display_startup_image("/home/pi/capstone/bgi.jpg", 0.01)
            time.sleep(1) # 버튼 채터링(떨림) 방지 및 사용자 인지 시간

            # --- "실행" 상태 ---
            # 메인 애플리케이션 함수를 호출합니다.
            # 이 함수는 정지 버튼이 눌리거나 종료될 때까지 끝나지 않습니다.
            main_application()

            # 메인 앱이 종료되면, 다시 루프의 처음(대기 상태)으로 돌아갑니다.

    except KeyboardInterrupt:
        print("\n프로그램을 완전히 종료합니다.")
    finally:
        # 프로그램이 어떤 이유로든 완전히 종료될 때 모든 리소스를 정리합니다.
        GPIO.cleanup()


