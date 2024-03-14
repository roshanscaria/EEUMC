import sounddevice as sd
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
import io
import warnings
import librosa
from transformers import pipeline

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Define audio settings
# RATE = 16000  # Sampling rate (16 kHz)
# DURATION = 5  # Record for 5 seconds
RATE = 16000#Changed for model validation
DURATION = 6
print("Recording...")

# Record audio
audio_data = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=1, dtype='int16')
sd.wait()

print("Recording stopped")

# Save recorded audio to a temporary WAV file
temp_wav_file = "recorded_audio9.wav"
sf.write(temp_wav_file, audio_data, RATE)

# Load the audio file using librosa
audio_file = "recorded_audio9.wav"
# audio_file = "/Users/roshanscaria/Desktop/Audio Emotion/TESTED AUDIOS/1001_DFA_DIS_XX.wav"
speech_array, sampling_rate = librosa.load(audio_file, sr=RATE)

# Preprocess the audio file
inputs = processor(speech_array, sampling_rate=RATE, return_tensors="pt", padding=True)

# Perform inference
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# Decode the logits to obtain the predicted transcription
predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentence = processor.batch_decode(predicted_ids)[0]

# Print the predicted transcription
print("Predicted Transcription:", predicted_sentence)




classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = predicted_sentence

model_outputs = classifier(sentences)
print(model_outputs[0][0]['label'])
# print(type(model_outputs[0][0]['label']))