from ai_edge_litert.interpreter import Interpreter

import soundfile as sf
import numpy as np
import resampy
import zipfile
import sounddevice as sd
import librosa
from scipy.io.wavfile import write

# --- 설정값 ---
RECORD_SECONDS = 0.975          # 녹음할 시간 (초)
ORIGINAL_RATE = 48000          # 마이크의 원본 샘플링 레이트 (장치에 따라 48000으로 변경)            # 녹음할 채널 수 (I2S 마이크는 보통 2)
TARGET_RATE = 16000             # 최종 WAV 파일의 샘플링 레이트
CHANNEL = 2             # 최종 WAV 파일의 채널 수 (모노)
FILENAME = "test4.wav"  # 저장할 파일 이름

def load_audio_file1(file_path, sample_rate= 16000):
    audio, sr=sf.read(file_path, dtype=np.float32)
    waveform = audio / np.max(np.abs(audio))
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != sample_rate:
      waveform = resampy.resample(waveform, sr, sample_rate)
    return waveform

def load_audio_file2(file_path, sample_rate= 16000):
    audio, fiile_sr=librosa.load(file_path, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    return audio

def real_record():
  print(sd.query_devices())
  print("recording...")

  recording = sd.rec(int(RECORD_SECONDS*ORIGINAL_RATE), samplerate=ORIGINAL_RATE, channels=CHANNEL, dtype='int32', device='hw:3,0')
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
waveform = load_audio_file2(FILENAME)
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
