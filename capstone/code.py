from ai_edge_litert.interpreter import Interpreter
import sys
import numpy as np
import sounddevice as sd
import scipy.signal
import collections
import zipfile
import RPi.GPIO as GPIO
import time
from PIL import Image, ImageDraw, ImageFont
import ST7735
import csv
import math

# --- PARAMETERS ---
VIBRATION_PIN = 13
CHUNK_DURATION = 2.0
SLIDE_STEP = 1.0
ORIGINAL_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
NORMALIZATION_BIT_DEPTH = 32
MAX_VAL_FOR_NORMALIZATION = 2 ** (NORMALIZATION_BIT_DEPTH - 1)
MIN_RMS_THRESHOLD = 0.01  # 반응할 최소 소리 크기 (소음 필터링)
DIRECTION_LABELS = ["왼쪽", "오른쪽"]
MODEL_INPUT_SAMPLES = 15600
SMOOTHING_WINDOW = 2
STOP_BUTTON_PIN = 17
START_BUTTON_PIN = 22

# --- DISPLAY SETUP ---
disp = ST7735.ST7735(
    port=0,
    cs=0,
    dc=25,
    backlight=24,
    rst=27,
    width=128,
    height=128,
    rotation=180,
    offset_left=2,
    offset_top=1,
    invert=False
)
font = ImageFont.load_default()
WIDTH = disp.width
HEIGHT = disp.height
scaled_width = 32
scaled_height = 32


# --- SYMBOL IMAGE PATHS ---
category_image = {
    '1': '/AI-sound-to-vibration-wearable/pi/capstone/1.jpg',
    '2': '/AI-sound-to-vibration-wearable/pi/capstone/2.jpg',
    '3': '/AI-sound-to-vibration-wearable/pi/capstone/3.jpg',
    '4': '/AI-sound-to-vibration-wearable/pi/capstone/4.jpg',
    '5': '/AI-sound-to-vibration-wearable/pi/capstone/5.jpg',
    '6': '/AI-sound-to-vibration-wearable/pi/capstone/6.jpg',
}

# --- TARGETED SOUNDS ---
sound_category = {
    317: '1', 318: '1', 319: '1', 390: '1', 391: '1', 392: '1', 393: '1', 394: '1', 396: '1',
    302: '2', 303: '2', 304: '2', 312: '2', 325: '2', 306: '2', 307: '2',
    420: '3', 421: '3', 422: '3', 423: '3', 424: '3', 428: '3', 429: '3', 430: '3',
    433: '4', 434: '4', 437: '4', 454: '4', 455: '4', 460: '4', 462: '4', 463: '4', 464: '4', 478: '4', 480: '4', 483: '4', 486: '4',
    313: '5', 475: '5', 355: '5',
    6: '6', 7: '6', 9: '6', 10: '6', 11: '6', 479: '6',
}

# --- VIBRATION PATTERNS ---
duty_dict = {
    '1': [20, 50, 30, 0], '2': [20, 20, 20, 20], '3': [50, 0, 50, 0],
    '4': [10, 20, 30, 40, 50], '5': [50, 40, 30, 20, 10], '6': [50, 50, 50, 50]
}


def display_startup_image(image_path, duration=3):
    try:
        startup_img = Image.open(image_path).resize((WIDTH, HEIGHT), Image.LANCZOS)
        disp.display(startup_img)
        time.sleep(duration)
    except Exception as e:
        print(f"시작 이미지 로드 실패: {e}")
        time.sleep(duration)

def calculate_peak_rms(audio_chunk, percentile=95):
    """CALCULATING PEAK RMS OF AUDIO CHUNKS"""
    abs_chunk = np.abs(audio_chunk)
    threshold = np.percentile(abs_chunk, percentile)
    peaks = abs_chunk[abs_chunk >= threshold]
    return np.mean(peaks) if peaks.size > 0 else 0

def main_application():
    """REALTIME RECORDING + INFERENCE"""
    print("메인 애플리케이션 시작. 리소스를 초기화합니다.")

    # GPIO SETUP
    GPIO.setup(STOP_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(VIBRATION_PIN, GPIO.OUT)

    # PWM SETUP
    pwm = GPIO.PWM(VIBRATION_PIN, 100)
    pwm.start(0)

    # MODEL LOADING
    model_path = '/home/pi/capstone/yamnet-tflite-classification-tflite-v1/1.tflite'
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    result_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
    labels_file = zipfile.ZipFile(model_path).open('yamnet_label_list.txt')
    class_names = [l.decode('utf-8').strip() for l in labels_file.readlines()]

    print(f"\n실시간 소리 감지 시작 (종료 버튼: GPIO {STOP_BUTTON_PIN})")
    buffer = np.zeros((int(CHUNK_DURATION * ORIGINAL_RATE), CHANNELS), dtype=np.int32)

    
    while True:
        if GPIO.input(STOP_BUTTON_PIN) == GPIO.LOW:
            time.sleep(1)
            break


        new_samples = int(SLIDE_STEP * ORIGINAL_RATE)
        recording = sd.rec(new_samples, samplerate=ORIGINAL_RATE, channels=CHANNELS, dtype='int32', device='hw:3,0')
        sd.wait()


        buffer = np.roll(buffer, -new_samples, axis=0)
        buffer[-new_samples:, :] = recording

        # --- DECIDING DIRECTION (VOLUME COMPARISON)  ---
        waveform_float_multi = buffer.astype(np.float32) / MAX_VAL_FOR_NORMALIZATION
        rms_values = [calculate_peak_rms(waveform_float_multi[:, i]) for i in range(CHANNELS)]
        loudest_channel_index = np.argmax(rms_values)
        max_rms = rms_values[loudest_channel_index]

        if max_rms < MIN_RMS_THRESHOLD:
            print("소리 감지 대기 중...", " " * 40, end='\r')
            continue

        detected_direction = DIRECTION_LABELS[loudest_channel_index]

        # --- AUDIO CLASSIFICATION ---
        waveform_mono = np.mean(waveform_float_multi, axis=1)

        # INPUT SAMPLE PROCESSING
        if ORIGINAL_RATE != TARGET_RATE:
            num_target = int(len(waveform_mono) * TARGET_RATE / ORIGINAL_RATE)
            waveform_resampled = scipy.signal.resample(waveform_mono, num_target)
        else:
            waveform_resampled = waveform_mono

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

        if len(scores.shape) == 2 and scores.shape[0] == 1:
            scores = scores[0]

      
        if len(scores) != len(class_names):
            print("모델 출력 shape가 비정상입니다. 입력 shape와 데이터 확인 필요.")
            continue

        top_class_index = np.argmax(scores)
        predicted_class = class_names[top_class_index]
        confidence = scores[top_class_index]

        # POST PROCESSING & RESULTING
        result_buffer.append(predicted_class)
        most_common = max(set(result_buffer), key=result_buffer.count)
        print(f" 방향:{detected_direction:<10} 최종 감지 소리: {most_common:<20} (신뢰도: {confidence:.2f})", end='\r\n')
        print(f"L채널: {rms_values[0]:.5f}         R채널: {rms_values[1]:.5f}")

        # --- OUTPUT ---
        idx = class_names.index(most_common)
        if idx in sound_category.keys() and confidence >= 0.4:
            # VISUAL FEEDBACK VIA DISPLAY
            try:
                final_image = Image.open("/home/pi/capstone/bgi.jpg").resize((WIDTH, HEIGHT), Image.LANCZOS)
                draw = ImageDraw.Draw(final_image)
            except Exception as e:
                print(f"배경 이미지 로드 실패: {e}")
                final_image = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
                draw = ImageDraw.Draw(final_image)

            category = sound_category[idx]
            img_path = category_image.get(category)

            if img_path:
                try:
                    icon_img = Image.open(img_path).resize((scaled_width, scaled_height), Image.LANCZOS)
                    icon_x = (WIDTH - scaled_width) // 2
                    icon_y = 80
                    final_image.paste(icon_img, (icon_x, icon_y), icon_img if icon_img.mode == 'RGBA' else None)
                except Exception as e:
                    print(f'아이콘 이미지 로드 실패: {e}')

            bbox = draw.textbbox((0, 0), most_common, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (WIDTH - text_width) // 2
            text_y = (HEIGHT - text_height) // 2 - 10
            draw.text((text_x, text_y), most_common, font=font, fill=(255, 255, 255))

            if not (0.99 < rms_values[0] / rms_values[1] < 1.01):
                if detected_direction == '왼쪽':
                    draw.text((icon_x - 20, icon_y + 8), "<-", font=font, fill=(255, 255, 0))
                else:
                    draw.text((icon_x + scaled_width + 10, icon_y + 8), "->", font=font, fill=(255, 255, 0))


            disp.display(final_image)

            # HAPTIC FEEDBACK VIA MOTOR
            for duty in duty_dict[category]:
                pwm.ChangeDutyCycle(duty)
                time.sleep(0.5)
            pwm.ChangeDutyCycle(0)

    print("\n메인 애플리케이션 리소스를 정리합니다.")
    GPIO.remove_event_detect(STOP_BUTTON_PIN)
    pwm.stop()


# ====================
# ======= MAIN =======
# ====================
if __name__ == "__main__":
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(START_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        display_startup_image("/home/pi/capstone/start_image.jpeg", 3)

        img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
        disp.display(img)



        while True:
            print("\n[대기 모드] 시작 버튼을 누르세요...")
            while GPIO.input(START_BUTTON_PIN) == GPIO.HIGH:
                time.sleep(0.01)

            print("시작 버튼 감지! 1초 후 감지를 시작합니다.")
            time.sleep(1)

            display_startup_image("/home/pi/capstone/bgi.jpg", 0.01)
            time.sleep(1) # 버튼 채터링(떨림) 방지 및 사용자 인지 시간
            main_application()
            img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
            disp.display(img)


    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        GPIO.cleanup()
