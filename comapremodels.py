# import os
# import pandas as pd
# import librosa
# import numpy as np
# import pickle
# from tensorflow.keras.models import model_from_json
# from collections import defaultdict

# # Load the models, scaler, and encoder
# # Loading the first model
# json_file = open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/CNN_model_new.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/CNN_model_weights_new.h5")

# # Loading the second model
# json_file2 = open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/WithoutNoise/CNN_model_WOnoise.json', 'r')
# loaded_model_json2 = json_file2.read()
# json_file2.close()
# loaded_model_WOnoise = model_from_json(loaded_model_json2)
# loaded_model_WOnoise.load_weights("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/WithoutNoise/CNN_model_weights_WOnoise.h5")

# # Loading scaler and encoder
# with open("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/scaler2.pickle", 'rb') as f:
#     scaler2 = pickle.load(f)

# with open("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/encoder2.pickle", 'rb') as f:
#     encoder2 = pickle.load(f)

# # Define functions for feature extraction and prediction
# def zcr(data, frame_length, hop_length):
#     zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
#     return np.squeeze(zcr)

# def rmse(data, frame_length=2048, hop_length=512):
#     rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
#     return np.squeeze(rmse)

# def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
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

# def get_predict_feat(path):
#     d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
#     res = extract_features(d)
#     result = np.array(res)
    
#     # Get the current size of the feature array
#     current_size = result.shape[0]
    
#     # Check if the current size is less than 2376
#     if current_size < 2376:
#         # Pad the feature array with zeros to match the desired shape (1, 2376)
#         padding_size = 2376 - current_size
#         result = np.pad(result, (0, padding_size))
    
#     # Reshape the feature array
#     result = np.reshape(result, newshape=(1, 2376))  # For 2.5 sec
#     i_result = scaler2.transform(result)
#     final_result = np.expand_dims(i_result, axis=2)
#     return final_result

# def prediction_with_probabilities(path, model1, model2):
#     res = get_predict_feat(path)
#     probabilities1 = model1.predict(res)
#     probabilities2 = model2.predict(res)
#     predicted_labels1 = encoder2.inverse_transform(probabilities1)
#     predicted_labels2 = encoder2.inverse_transform(probabilities2)
#     return predicted_labels1[0][0], predicted_labels2[0][0]

# # Path to the folder containing audio files
# folder_path = "/Users/roshanscaria/Desktop/Audio Emotion/TESTED AUDIOS"

# # List all files in the folder
# audio_files = os.listdir(folder_path)

# # List to store predictions
# predictions_summary = []

# # Iterate over each audio file
# for file_name in audio_files:
#     # Construct the full path to the audio file
#     audio_path = os.path.join(folder_path, file_name)
    
#     # Perform prediction for the current audio file
#     label1, label2 = prediction_with_probabilities(audio_path, loaded_model, loaded_model_WOnoise)
    
#     # Append the predictions to the list
#     predictions_summary.append((file_name, label1, label2))

# # Convert predictions_summary to DataFrame
# summary_df = pd.DataFrame(predictions_summary, columns=['Filename', 'Predicted_Label_Model1', 'Predicted_Label_Model2'])

# # Display the summary DataFrame
# print(summary_df)




import os
import pandas as pd
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json

# Load the first model
json_file = open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/CNN_model_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/CNN_model_weights_new.h5")

# Load the second model
json_file2 = open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/WithoutNoise/CNN_model_WOnoise.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
loaded_model_WOnoise = model_from_json(loaded_model_json2)
loaded_model_WOnoise.load_weights("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/WithoutNoise/CNN_model_weights_WOnoise.h5")

# Load the third model
json_file3 = open('/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/WithoutNoise/CNN_model_WOnois_new.json', 'r')
loaded_model_json3 = json_file3.read()
json_file3.close()
loaded_model_WOnoise_new = model_from_json(loaded_model_json3)
loaded_model_WOnoise_new.load_weights("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/WithoutNoise/CNN_model_weights_WOnois_new.h5")

# Load scaler and encoder
with open("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/scaler2.pickle", 'rb') as f:
    scaler2 = pickle.load(f)

with open("/Users/roshanscaria/Desktop/Audio Emotion/TestAudioModelFiles/encoder2.pickle", 'rb') as f:
    encoder2 = pickle.load(f)

# Define functions for feature extraction and prediction
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    
    # Get the current size of the feature array
    current_size = result.shape[0]
    
    # Check if the current size is less than 2376
    if current_size < 2376:
        # Pad the feature array with zeros to match the desired shape (1, 2376)
        padding_size = 2376 - current_size
        result = np.pad(result, (0, padding_size))
    
    # Reshape the feature array
    result = np.reshape(result, newshape=(1, 2376))  # For 2.5 sec
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

def prediction_with_probabilities(path, model):
    res = get_predict_feat(path)
    probabilities = model.predict(res)
    predicted_labels = encoder2.inverse_transform(probabilities)
    return predicted_labels[0][0]

# Path to the folder containing audio files
folder_path = "/Users/roshanscaria/Desktop/Audio Emotion/TESTED AUDIOS"

# List all files in the folder
audio_files = os.listdir(folder_path)

# List to store predictions
predictions_summary = []

# Iterate over each audio file
for file_name in audio_files:
    # Construct the full path to the audio file
    audio_path = os.path.join(folder_path, file_name)
    
    # Perform prediction for the current audio file using all three models
    label1 = prediction_with_probabilities(audio_path, loaded_model)
    label2 = prediction_with_probabilities(audio_path, loaded_model_WOnoise)
    label3 = prediction_with_probabilities(audio_path, loaded_model_WOnoise_new)
    
    # Append the predictions to the list
    predictions_summary.append((file_name, label1, label2, label3))

# Convert predictions_summary to DataFrame
summary_df = pd.DataFrame(predictions_summary, columns=['Filename', 'Predicted_Label_Model1', 'Predicted_Label_Model2', 'Predicted_Label_Model3'])

# Display the summary DataFrame
print(summary_df)


# Initialize an empty list to store filtered predictions
ravdess_predictions = []

# Iterate through each row in summary_df
for index, row in summary_df.iterrows():
    # Extract the file name from the row
    file_name = row['Filename']
    
    # Check if the file name starts with '03'
    if file_name.startswith('03'):
        # Add the row to the list of filtered predictions
        ravdess_predictions.append(row)

# Create a new DataFrame for Ravdess predictions
ravdess_df = pd.DataFrame(ravdess_predictions)
print("\n\n\n\nRAVDEES\n\n\n")
# Display the Ravdess DataFrame
print(ravdess_df)



# Initialize an empty list to store filtered predictions for CREMAD
CREMAD_predictions = []

# Iterate through each row in summary_df
for index, row in summary_df.iterrows():
    # Extract the file name from the row
    file_name = row['Filename']
    
    # Check if the file name starts with a four-digit number
    if file_name[:4].isdigit():
        # Add the row to the list of filtered predictions for CREMAD
        CREMAD_predictions.append(row)

# Create a new DataFrame for CREMAD predictions
CREMAD_df = pd.DataFrame(CREMAD_predictions)
print("\n\n\n\n CREMAD \n\n")
# Display the CREMAD DataFrame
print(CREMAD_df)


# Initialize an empty list to store filtered predictions for SAVEE
SAVEE_predictions = []

# Iterate through each row in summary_df
for index, row in summary_df.iterrows():
    # Extract the file name from the row
    file_name = row['Filename']
    
    # Check if the file name starts with two letters followed by an underscore
    if len(file_name) >= 3 and file_name[:2].isalpha() and file_name[1] != "Z" and file_name[2] == '_':
        # Add the row to the list of filtered predictions for SAVEE
        SAVEE_predictions.append(row)
print("\n\n\n\n SAVEE\n\n\n")
# Create a new DataFrame for SAVEE predictions
SAVEE_df = pd.DataFrame(SAVEE_predictions)

# Display the SAVEE DataFrame
print(SAVEE_df)


# Initialize an empty list to store filtered predictions for TESS
TESS_predictions = []

# Iterate through each row in summary_df
for index, row in summary_df.iterrows():
    # Extract the file name from the row
    file_name = row['Filename']
    
    # Check if the file name starts with 'OAF' or 'YAF'
    if file_name.startswith(('OAF', 'YAF')):
        # Add the row to the list of filtered predictions for TESS
        TESS_predictions.append(row)

# Create a new DataFrame for TESS predictions
TESS_df = pd.DataFrame(TESS_predictions)
print("\n\n\nTESS\n\n\n")
# Display the TESS DataFrame
print(TESS_df)


# Combine all filenames from ravdess_df, CREMAD_df, and TESS_df
excluded_filenames = set(ravdess_df['Filename']).union(set(CREMAD_df['Filename'])).union(set(TESS_df['Filename'])).union(set(SAVEE_df['Filename']))

# Filter out rows where the filename is not in excluded_filenames
other_predictions = summary_df[~summary_df['Filename'].isin(excluded_filenames)]
print("\n\n\n")
# Display the DataFrame for other predictions
print(other_predictions)
