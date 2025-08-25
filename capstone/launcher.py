# 파일명: launcher.py
import RPi.GPIO as GPIO
import subprocess
import time
#from PIL import Image, ImageDraw, ImageFont
#import ST7735
'''
disp = ST7735.ST7735(
    port=0,           # SPI 포트 (일반적으로 0)
    cs=0,             # SPI CS (일반적으로 0)
    dc=25,            # Data/Command 핀 (예: GPIO 24)
    backlight=24,     # 백라이트 핀 (없으면 None)
    rst=27,           # Reset 핀 (예: GPIO 25)
    width=128,        # 디스플레이 가로 해상도
    height=128,       # 디스플레이 세로 해상도
    rotation=0,
    offset_left=2,
    offset_top=1,     # 회전 각도 (필요에 따라 0, 90, 180, 270)
    invert=False      # 색상 반전 여부
)

WIDTH = disp.width
HEIGHT = disp.height
'''
# 사용할 GPIO 핀 번호 (BCM 모드)
BUTTON_PIN = 17

# 실행할 메인 스크립트의 전체 경로
MAIN_SCRIPT_PATH = '/home/pi/capstone/aa.py' # 실제 경로로 수정하세요.

# GPIO 설정

'''
img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
'''
print("런처 스크립트가 시작되었습니다.")
'''
text = "TEAM S.O.S."
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (WIDTH - text_width) // 2
y = (HEIGHT - text_height) // 2
draw.text((x, y), text, font=font, fill=(255, 255, 255))
disp.display(img)
time.sleep(3)
'''

try:
    # 프로그램이 종료되지 않도록 무한 루프를 사용합니다.
    while True:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print("\n[대기 모드] 버튼 입력을 기다립니다...")
        #draw.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=(0, 0, 0))
        #disp.display(img)


        while GPIO.input(BUTTON_PIN):
            time.sleep(0.01)

        print(f"버튼 감지! 1초 후 '{MAIN_SCRIPT_PATH}'를 실행합니다.")
        time.sleep(0.2) # 버튼 채터링(떨림) 방지 및 사용자 인지 시간
        GPIO.cleanup()
        time.sleep(2)
        # 2. 메인 스크립트를 실행하고, 이 스크립트가 끝날 때까지 여기서 대기합니다.
        try:
            # subprocess.run()은 main.py가 종료될 때까지 다음으로 진행되지 않습니다.
            subprocess.run(['python3', MAIN_SCRIPT_PATH], check=True)
        except subprocess.CalledProcessError as e:
            # 메인 스크립트 내부에서 오류가 발생하여 비정상 종료된 경우
            print(f"메인 스크립트에서 오류가 발생했습니다: {e}")
        except KeyboardInterrupt:
            # 이 부분은 거의 호출되지 않으나 안전장치로 둡니다.
            # main.py에서 Ctrl+C를 누르면 main.py가 인터럽트를 받고 종료됩니다.
            pass

        # 3. 메인 스크립트가 종료되면 자동으로 이 부분이 실행됩니다.
        print("메인 스크립트가 종료되었습니다. 런처가 다시 대기 모드로 전환됩니다.")

except KeyboardInterrupt:
    print("\n런처 프로그램을 완전히 종료합니다.")
finally:
    GPIO.cleanup()


