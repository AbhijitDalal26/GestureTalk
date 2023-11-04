import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Get the absolute path of the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full file path for the 'data' directory
DATA_DIR = os.path.join(script_directory, 'data')

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):  # Only consider directories
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

# Save the preprocessed data using pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# The file is automatically closed after exiting the 'with' block