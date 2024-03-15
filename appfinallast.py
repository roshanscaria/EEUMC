from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import pickle
import sounddevice as sd
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import pipeline
from tensorflow.keras.models import model_from_json
import soundfile as sf
from scipy.signal import resample

app = Flask(__name__)

# Define audio settings
RATE = 16000  # Sampling rate (22.05 kHz)
DURATION = 5  # Record for 2.5 seconds

# Define the directory to save recordings
RECORDINGS_DIR = "/Users/roshanscaria/Desktop/Audio Emotion/TESTUIwithall/static/recordings"

# Load model and processor for audio processing
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Define the sentiment analysis pipeline
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

emotions1= {1:'angry', 2:'disgust', 3:'neutral', 4:'happy', 5:'calm', 6:'sad', 7:'surprise',8:'fear'}

# Load the CNN model from JSON and weights
json_file = open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/CNN_model_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/CNN_model_weights_new.h5")
print("Loaded model from disk")

with open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/scalernew.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

# Loading encoder
with open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/encodernew.pickle', 'rb') as f:
    encoder2 = pickle.load(f)


# def zcr(data, frame_length, hop_length):
#     zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
#     return np.squeeze(zcr)


# def rmse(data, frame_length=2048, hop_length=512):
#     rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
#     return np.squeeze(rmse)


# def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
#     mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
#     return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


# def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
#     result = np.array([])

#     result = np.hstack((result,
#                         zcr(data, frame_length, hop_length),
#                         rmse(data, frame_length, hop_length),
#                         mfcc(data, sr, frame_length, hop_length)
#                         ))
#     return result
    

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
# def rmse(data,frame_length=2048,hop_length=512):
#     rmse=librosa.feature.rms(data,frame_length=frame_length,hop_length=hop_length)
#     return np.squeeze(rmse)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


# def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
#     mfcc=librosa.feature.mfcc(data,sr=sr)
#     return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)
# def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
#     mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
#     return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])

    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result


# def get_predict_feat(path):
#     d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
#     res = extract_features(d)
#     result = np.array(res)
#     result = np.reshape(result, newshape=(1, 2376))  # For 2.5 sec
#     i_result = scaler2.transform(result)
#     final_result=np.expand_dims(i_result, axis=2)
    
#     return final_result 

def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))#For 2.5 sec
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    
    return final_result 


# def predictionall(path1):
#     # Load the recorded audio file
#     speech_array, _ = librosa.load(path1, sr=RATE)
#     res = get_predict_feat(path1)
#     predictions = loaded_model.predict(res)

#     # Getting the predicted probabilities for all classes
#     probabilities = predictions[0]

#     # Printing all the probabilities and their corresponding emotions
#     for i, prob in enumerate(probabilities):
#         emotion = emotions1[i + 1]  # Emotion corresponding to index i+1
#         print(f"Emotion: {emotion}, Probability: {prob}")

#     # Retrieving the predicted emotion based on the highest probability
#     predicted_index = np.argmax(probabilities)
#     predicted_emotion = emotions1[predicted_index + 1]
#     print("Predicted Emotion:", predicted_emotion)
#To see all the probabilties
# Function to resample audio to 16 kHz
def resample_audio_to_16k(input_file, output_file):
    # Load the audio file
    y, sr = sf.read(input_file)

    # Check if the sampling rate is already 16 kHz
    if sr == 16000:
        print("Sampling rate is already 16 kHz. No need to resample.")
        return input_file  # Return the original file path

    # Resample the audio to the target sample rate (16 kHz)
    target_sr = 16000
    y_resampled = resample(y, int(len(y) * target_sr / sr))

    # Save the resampled audio to a new file
    sf.write(output_file, y_resampled, target_sr)
    return output_file


def predictionall(path1):
    res = get_predict_feat(path1)
    predictions = loaded_model.predict(res)

    # Getting the predicted probabilities for all classes
    probabilities = predictions[0]

    # Printing all the probabilities and their corresponding emotions
    for i, prob in enumerate(probabilities):
        emotion = emotions1[i+1]  # Emotion corresponding to index i+1
        print(f"Emotion: {emotion}, Probability: {prob}")

    # Retrieving the predicted emotion based on the highest probability
    predicted_index = np.argmax(probabilities)
    predicted_emotion = emotions1[predicted_index + 1]
    print("Predicted Emotion:", predicted_emotion)



    speech_array, _ = librosa.load(path1, sr=RATE)
    # Perform transcription
    inputs = processor(speech_array, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = wav2vec_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_transcription = processor.batch_decode(predicted_ids)[0]

    # Perform sentiment analysis on the predicted transcription
    sentiment_analysis = classifier(predicted_transcription)
    predicted_sentiment = sentiment_analysis[0][0]['label']

    return predicted_emotion, predicted_transcription, predicted_sentiment


@app.route('/')
def index():
    return render_template('page.html')

# Process text feedback
@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        text = request.json['text']
        # Get model prediction
        model_outputs = classifier(text)
        # Extract the predicted label from the first output
        predicted_label = model_outputs[0][0]['label']
        return jsonify({'sentiment': predicted_label})


@app.route('/process_audio', methods=['POST'])
def process_audio():
    if request.method == 'POST':
        try:
            print("Recording...")
            audio_data = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=1, dtype='int16')
            sd.wait()
            print("Recording stopped")

            # Save recorded audio to a temporary WAV file
            temp_wav_file = os.path.join(RECORDINGS_DIR, "recorded_audio.wav")
            sf.write(temp_wav_file, audio_data, RATE)
            # temp_wav_file = "/Users/roshanscaria/Desktop/Audio Emotion/TESTED AUDIOS/1001_DFA_DIS_XX.wav"
            

            # Check if resampling is necessary and resample audio if needed
            resampled_audio_file = resample_audio_to_16k(temp_wav_file, temp_wav_file + "_16k.wav")


            # Call the prediction function
            predicted_emotion, predicted_transcription, predicted_sentiment = predictionall(temp_wav_file)

            result = {
                'emotion': predicted_emotion,
                'transcription': predicted_transcription,
                'sentiment': predicted_sentiment
            }
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
