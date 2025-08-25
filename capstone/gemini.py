# 필요한 라이브러리만 남겨둡니다.
import sys
import RPi.GPIO as GPIO
import time
from PIL import Image, ImageDraw, ImageFont
import ST7735

# --- 설정값 (디스플레이, GPIO 관련은 유지) ---
VIBRATION_PIN = 13
STOP_BUTTON_PIN = 17
START_BUTTON_PIN = 22
WIDTH = 128
HEIGHT = 128
scaled_width = 32
scaled_height = 32

# 디스플레이 객체 생성
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

# 이미지 및 카테고리 정보는 그대로 사용
category_image = {
    '1': '/home/pi/capstone/1.jpg', '2': '/home/pi/capstone/2.jpg',
    '3': '/home/pi/capstone/3.jpg', '4': '/home/pi/capstone/4.jpg',
    '5': '/home/pi/capstone/5.jpg', '6': '/home/pi/capstone/6.jpg',
}

sound_category = {
    317: '1', 318: '1', 319: '1', 390: '1', 391: '1', 392: '1', 393: '1', 394: '1', 396: '1',
    302: '2', 303: '2', 304: '2', 312: '2', 325: '2', 306: '2', 307: '2',
    420: '3', 421: '3', 422: '3', 423: '3', 424: '3', 428: '3', 429: '3', 430: '3',
    433: '4', 434: '4', 437: '4', 454: '4', 455: '4', 460: '4', 462: '4', 463: '4', 464: '4', 478: '4', 480: '4', 483: '4', 486: '4',
    313: '5', 475: '5', 355: '5',
    6: '6', 7: '6', 9: '6', 10: '6', 11: '6', 479: '6',
}

duty_dict = {
    '1': [20, 50, 30, 0], '2': [20, 20, 20, 20], '3': [50, 0, 50, 0],
    '4': [10, 20, 30, 40, 50], '5': [50, 40, 30, 20, 10], '6': [50, 50, 50, 50]
}

# 폰트 로드
font = ImageFont.load_default()

def display_startup_image(image_path, duration=3):
    """시작화면 이미지를 디스플레이에 표시하는 함수"""
    try:
        startup_img = Image.open(image_path).resize((WIDTH, HEIGHT), Image.LANCZOS)
        disp.display(startup_img)
        time.sleep(duration)
    except Exception as e:
        print(f"시작화면 이미지 로드 실패: {e}")
        time.sleep(duration)

def main_application():
    """(테스트용) 이미지 출력 및 진동 테스트 로직"""
    print("이미지 출력 테스트 시작. (종료 버튼: GPIO 17)")

    # GPIO 설정
    GPIO.setup(STOP_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(VIBRATION_PIN, GPIO.OUT)

    # PWM 객체 생성
    pwm = GPIO.PWM(VIBRATION_PIN, 100)
    pwm.start(0)

    # --- 테스트할 데이터 정의 ---
    # (소리 ID, '표시할 텍스트', '방향', [왼쪽 소리크기, 오른쪽 소리크기])
    test_cases = [
        {'idx': 317, 'most_common': 'Police car (siren)', 'direction': '왼쪽', 'rms': [0.8, 0.2]},
        {'idx': 302, 'most_common': 'Car horn', 'direction': '오른쪽', 'rms': [0.2, 0.8]},
        {'idx': 420, 'most_common': 'Explosion', 'direction': '오른쪽', 'rms': [0.3, 0.9]},
        {'idx': 463, 'most_common': 'Crash', 'direction': '왼쪽', 'rms': [0.9, 0.1]},
        {'idx': 313, 'most_common': 'Reversing beeps', 'direction': '오른쪽', 'rms': [0.4, 0.7]},
        {'idx': 11, 'most_common': 'Screaming', 'direction': '왼쪽', 'rms': [0.8, 0.3]},
        # 중앙에서 소리가 나는 경우 테스트 (화살표 표시 안됨)
        {'idx': 394, 'most_common': 'Fire alarm', 'direction': '왼쪽', 'rms': [0.8, 0.8]},
    ]

    # --- 테스트 루프 ---
    for case in test_cases:
        # 종료 버튼이 눌리면 테스트 중단
        if GPIO.input(STOP_BUTTON_PIN) == GPIO.LOW:
            print("테스트를 중단합니다.")
            time.sleep(1) # 버튼 디바운싱
            break

        print(f"테스트 중: {case['most_common']} ({case['direction']})")

        # 시뮬레이션 데이터 추출
        idx = case['idx']
        most_common = case['most_common']
        detected_direction = case['direction']
        rms_values = case['rms']

        # --- 기존 디스플레이 로직 (그대로 사용) ---
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
                icon_y = 72
                final_image.paste(icon_img, (icon_x, icon_y), icon_img if icon_img.mode == 'RGBA' else None) # 투명 배경 지원
            except Exception as e:
                print(f'아이콘 이미지 로드 실패: {e}')

        bbox = draw.textbbox((0, 0), most_common, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (WIDTH - text_width) // 2
        text_y = (HEIGHT - text_height) // 2 - 10
        draw.text((text_x, text_y), most_common, font=font, fill=(255, 255, 255))

        # rms_values[1]이 0이 되는 경우를 대비하여 작은 값(1e-6)을 더해줌
        if not (0.9 < rms_values[0] / (rms_values[1] + 1e-6) < 1.1):
            if detected_direction == '왼쪽':
                draw.text((icon_x - 20, icon_y + 8), "<-", font=font, fill=(255, 255, 0))
            else:
                draw.text((icon_x + scaled_width + 10, icon_y + 8), "->", font=font, fill=(255, 255, 0))

        disp.display(final_image)

        # 진동 패턴 테스트
        for duty in duty_dict[sound_category[idx]]:
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.2) # 진동 테스트는 짧게
        pwm.ChangeDutyCycle(0)

        # 다음 테스트 전 3초 대기
        time.sleep(3)

    # --- 테스트 완료 후 리소스 정리 ---
    print("\n테스트 완료. 리소스를 정리합니다.")
    pwm.stop()
    # STOP_BUTTON_PIN에 대한 이벤트 감지는 이 함수에서 설정했으므로 여기서 제거하지 않아도 됩니다.
    # main_application 함수가 끝나면 자동으로 대기 상태로 돌아갑니다.

# =============================================
# === 프로그램의 메인 진입점 (State Machine) ===
# =============================================
if __name__ == "__main__":
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(START_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        display_startup_image("/home/pi/capstone/start_image.jpeg", 3)

        # 시작 전 화면을 검은색으로 초기화
        img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
        disp.display(img)

        while True:
            print("\n[대기 모드] 시작 버튼을 누르세요...")
            # 대기 상태에서 시작 버튼을 기다림
            while GPIO.input(START_BUTTON_PIN) == GPIO.HIGH:
                time.sleep(0.01)

            print("시작 버튼 감지! 1초 후 테스트를 시작합니다.")
            time.sleep(1) # 버튼 채터링 방지

            # --- "실행" 상태 ---
            # 테스트용으로 수정된 main_application 함수를 호출합니다.
            main_application()

    except KeyboardInterrupt:
        print("\n프로그램을 완전히 종료합니다.")
    finally:
        GPIO.cleanup()