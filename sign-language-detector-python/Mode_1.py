import cv2
import os
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

script_directory = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_directory, 'model.p')
model_dict = pickle.load(open(model_file_path, 'rb'))
model = model_dict['model']

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'thumbs up', 27: 'thumbs down', 28: ' ', 29: 'Left', 30: 'Very Nice'}

prediction_interval = 2  # 2 seconds interval for predictions
prev_time = time.time()  # Initialize previous time for timing predictions

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        if time.time() - prev_time > prediction_interval:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

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

                prediction = model.predict([np.asarray(data_aux)])
                predicted_alphabet = labels_dict[int(prediction[0])]

                if predicted_alphabet != 'thumbs up':
                    print("Predicted Letter:", predicted_alphabet)  # Print predicted letter
                    engine.say(predicted_alphabet)  # Speak the predicted letter
                    engine.runAndWait()  # Wait for the speech to finish
                    prev_time = time.time()  # Update previous time

        cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()