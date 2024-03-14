import soundfile as sf
from scipy.signal import resample

# Load the audio file
input_file = "/Users/roshanscaria/Desktop/Audio Emotion/TESTED AUDIOS/1001_DFA_DIS_XX.wav"
y, sr = sf.read(input_file)

print(sr)

# Resample the audio to the target sample rate
target_sr = 16000
y_resampled = resample(y, int(len(y) * target_sr / sr))

# Save the resampled audio to a new file
output_file = "output_16khz.wav"
sf.write(output_file, y_resampled, target_sr)
