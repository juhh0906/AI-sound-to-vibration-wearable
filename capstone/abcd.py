from ai_edge_litert.interpreter import Interpreter

import librosa
import numpy as np
import zipfile
import sounddevice as sd
from scipy.io.wavfile import write

# --- 설정값 ---
RECORD_SECONDS = 0.975          # 녹음할 시간 (초)
ORIGINAL_RATE = 48000          # 마이크의 원본 샘플링 레이트 (장치에 따라 48000으로 변경)            # 녹음할 채널 수 (I2S 마이크는 보통 2)
TARGET_RATE = 16000             # 최종 WAV 파일의 샘플링 레이트
CHANNEL = 1             # 최종 WAV 파일의 채널 수 (모노)
#FORMAT = pyaudio.paFloat32       # 16비트 오디오 포맷
#CHUNK = 1024                    # 한 번에 읽을 프레임 수
FILENAME = "test4.wav"  # 저장할 파일 이름

def load_audio_file(file_path, sample_rate= 16000):
    audio, fiile_sr=librosa.load(file_path, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    return audio

'''def record_and_create_wav(filename):
    """
    지정된 시간 동안 오디오를 녹음하고, 16kHz 모노 WAV 파일로 변환하여 저장합니다.
    """
    # 1. PyAudio 초기화 및 스트림 열기
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=RECORD_CHANNELS,
                        rate=ORIGINAL_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print(f"{RECORD_SECONDS}초 동안 녹음을 시작합니다...")

    # 2. 필요한 시간만큼 데이터 녹음
    frames = []
    num_chunks_to_record = int(ORIGINAL_RATE / CHUNK * RECORD_SECONDS)
    for _ in range(num_chunks_to_record):
        data = stream.read(CHUNK)
        frames.append(data)

    print("녹음 완료.")

    # 3. 스트림 정지 및 리소스 해제
    stream.stop_stream()
    stream.close()
    audio.terminate()

   

    # 8. WAV 파일로 저장
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(TARGET_RATE)
        wf.writeframes(b''.join(frames))

    print(f"성공적으로 '{filename}' 파일을 저장했습니다 ")'''

def real_record():
  print(sd.query_devices())
  print("recording...")
  
  recording = sd.rec(int(RECORD_SECONDS*ORIGINAL_RATE), samplerate=ORIGINAL_RATE, channels=CHANNEL, dtype='int32', device='hw:1,0')
  sd.wait()
  
  print("recording completed")
  
  write(FILENAME, ORIGINAL_RATE, recording)
  
  


model_path = '/home/pi/capstone/yamnet-tflite-classification-tflite-v1/1.tflite'
interpreter = Interpreter(model_path)

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']

# Input: 0.975 seconds of silence as mono 16 kHz waveform samples.
#record_and_create_wav(FILENAME)
real_record()
waveform = load_audio_file(FILENAME)
print(waveform.shape)  # Should print (15600,)

interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=True)
interpreter.allocate_tensors()
interpreter.set_tensor(waveform_input_index, waveform)
interpreter.invoke()
scores = interpreter.get_tensor(scores_output_index)
print(scores.shape)  # Should print (1, 521)

top_class_index = scores.argmax()
labels_file = zipfile.ZipFile(model_path).open('yamnet_label_list.txt')
labels = [l.decode('utf-8').strip() for l in labels_file.readlines()]
print(len(labels))  # Should print 521
print(labels[top_class_index])  # Should print 'Silence'.
