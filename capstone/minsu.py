from ai_edge_litert.interpreter import Interpreter
import numpy as np
import sounddevice as sd
import scipy.signal
import csv
import collections
import zipfile
import math

# --- 설정값 ---
CHUNK_DURATION = 2.0
SLIDE_STEP = 1.0
ORIGINAL_RATE = 48000
TARGET_RATE = 16000
CHANNELS = 2
MIN_ENERGY_THRESHOLD = 0.01
NORMALIZATION_BIT_DEPTH = 24
MAX_VAL_FOR_NORMALIZATION = 2**(NORMALIZATION_BIT_DEPTH - 1)
#TFLITE_MODEL_PATH = "yamnet_quant.tflite"
MODEL_INPUT_SAMPLES = 15600
SMOOTHING_WINDOW = 2
MIC_DISTANCE_CM = 30.0        # 마이크 사이의 거리 (cm) - **실제 값으로 수정 필수**
SPEED_OF_SOUND_CM_S = 34300   # 소리의 속도 (cm/s)
MIC_DISTANCE_M = MIC_DISTANCE_CM / 100.0
MAX_TDOA = MIC_DISTANCE_M / (SPEED_OF_SOUND_CM_S / 100.0)


'''zip_path = 'YAMNET-Sound-Classification-python-main.zip'
csv_filename = 'yamnet_class_map.csv'''

def calculate_angle(tdoa_seconds):
    """시간 차이(TDOA)를 바탕으로 각도를 계산합니다."""
    # ratio가 -1과 1 사이를 벗어나지 않도록 클리핑
    ratio = max(-1.0, min(1.0, tdoa_seconds / MAX_TDOA))
    angle_rad = math.asin(ratio)
    return math.degrees(angle_rad)
#class_names = class_names_from_zip(zip_path, csv_filename)

model_path = '/home/pi/capstone/yamnet-tflite-classification-tflite-v1/1.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
result_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
labels_file = zipfile.ZipFile(model_path).open('yamnet_label_list.txt')
class_names = [l.decode('utf-8').strip() for l in labels_file.readlines()]

try:
    print("\n실시간 소리 감지 시작 (종료: Ctrl+C)")
    buffer = np.zeros((int(CHUNK_DURATION * ORIGINAL_RATE), CHANNELS), dtype=np.int32)
    while True:
        slide_samples = int(SLIDE_STEP * ORIGINAL_RATE)
        recording = sd.rec(slide_samples, samplerate=ORIGINAL_RATE, channels=CHANNELS, dtype='int32', device='hw:3,0')
        sd.wait()
        '''buffer = np.roll(buffer, -new_samples)
        buffer[-new_samples:] = recording.flatten()

        # 정규화
        waveform_float = buffer.astype(np.float32) / MAX_VAL_FOR_NORMALIZATION
        waveform_mono = waveform_float.flatten()

        # 에너지 임계값 검사
        peak_energy = np.max(np.abs(waveform_mono))
        if peak_energy < MIN_ENERGY_THRESHOLD:
            print(f"소리 감지 대기 중... (에너지: {peak_energy:.4f})", end='\r')
            continue

        # 피크 정규화
        waveform_scaled = waveform_mono / peak_energy'''
        buffer = np.roll(buffer, -slide_samples, axis=0)
        buffer[-slide_samples:, :] = recording
        
        left_channel = buffer[:, 0]
        right_channel = buffer[:, 1]
        '''
        corr = scipy.signal.correlate(left_channel, right_channel, mode='full')
        delay_index = np.argmax(corr)
        center_index = (len(corr) - 1) / 2
        delay_samples = delay_index - center_index
        delay_seconds = delay_samples / ORIGINAL_RATE
        '''
        #angle = calculate_angle(delay_seconds)        

            # 2. 에너지 임계값 검사 (모노 변환 후)
        waveform_float = buffer.astype(np.float32) / MAX_VAL_FOR_NORMALIZATION
        waveform_mono = waveform_float.flatten()
        peak_energy = np.max(np.abs(waveform_mono))

        if peak_energy < MIN_ENERGY_THRESHOLD:
            print(f"소리 감지 대기 중... (에너지: {peak_energy:.4f})", end='\r')
            continue

            # ============================================
            # 3. 방향 탐지 (TDOA) - 스테레오 데이터 사용
            # ============================================
        # 리샘플링
            # 리샘플링 (48000Hz -> 16000Hz)
        num_target = int(len(waveform_mono) * TARGET_RATE / ORIGINAL_RATE)
        waveform_resampled = scipy.signal.resample(waveform_mono, num_target)

            # 입력 길이 맞춤 (패딩/자르기)
        if len(waveform_resampled) > MODEL_INPUT_SAMPLES:
            waveform_resampled = waveform_resampled[:MODEL_INPUT_SAMPLES]
        elif len(waveform_resampled) < MODEL_INPUT_SAMPLES:
            padding = np.zeros(MODEL_INPUT_SAMPLES - len(waveform_resampled))
            waveform_resampled = np.concatenate([waveform_resampled, padding])
            
        # shape 오류 방지: 반드시 1차원으로!
        waveform_resampled = np.squeeze(waveform_resampled)
        print(waveform_resampled.shape)  # (15600,) 확인

        # TFLite 입력 shape 확인
        input_shape = tuple(input_details[0]['shape'])
        if input_shape == (15600,):
            interpreter.set_tensor(input_details[0]['index'], waveform_resampled.astype(np.float32))
        elif input_shape == (1, 15600):
            input_tensor = np.expand_dims(waveform_resampled, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
        else:
            raise ValueError(f"예상치 못한 입력 shape: {input_shape}")            

            # 모델 추론
        #interpreter.set_tensor(input_details[0]['index'], np.expand_dims(waveform_resampled, axis=0))
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]['index'])[0]


            # 가장 높은 점수의 클래스 찾기
        top_class_index = np.argmax(scores)
        predicted_class = class_names[top_class_index]

            # 결과 스무딩
        result_buffer.append(predicted_class)
        most_common = max(set(result_buffer), key=list(result_buffer).count)

            # 5. 최종 결과 출력
        print(f"감지된 소리: {most_common:<20s} (에너지: {peak_energy:.3f}) ", end='\r')    
        #print(f"방향: {angle:>6.1f}°  |  감지된 소리: {most_common:<20s} (에너지: {peak_energy:.3f}) ", end='\r')

        '''
        if ORIGINAL_RATE != TARGET_RATE:
            num_target = int(len(waveform_scaled) * TARGET_RATE / ORIGINAL_RATE)
            waveform_resampled = scipy.signal.resample(waveform_scaled, num_target)
        else:
            waveform_resampled = waveform_scaled

        # 입력 길이 맞춤(패딩/자르기)
        if len(waveform_resampled) > MODEL_INPUT_SAMPLES:
            waveform_resampled = waveform_resampled[:MODEL_INPUT_SAMPLES]
        elif len(waveform_resampled) < MODEL_INPUT_SAMPLES:
            padding = np.zeros(MODEL_INPUT_SAMPLES - len(waveform_resampled))
            waveform_resampled = np.concatenate([waveform_resampled, padding])

        # shape 오류 방지: 반드시 1차원으로!
        waveform_resampled = np.squeeze(waveform_resampled)
        print(waveform_resampled.shape)  # (15600,) 확인

        # TFLite 입력 shape 확인
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

        # 정상적으로 클래스 개수와 일치하는지 확인
        if len(scores) != len(class_names):
            print("모델 출력 shape가 비정상입니다. 입력 shape와 데이터 확인 필요.")
            continue

        top_class_index = np.argmax(scores)
        predicted_class = class_names[top_class_index]
        confidence = scores[top_class_index]

        # 후처리(다수결 스무딩)
        result_buffer.append(predicted_class)
        most_common = max(set(result_buffer), key=result_buffer.count)
        print(f"최종 감지 소리: {most_common:<20} (신뢰도: {confidence:.2f})", end='\r')'''

except KeyboardInterrupt:
    print("\n프로그램 종료.")