import os
import cv2

BASE_DIR = 'D:/Collage Project/American Sign Language/ASL/sign-language-detector-python'
DATA_DIR = os.path.join(BASE_DIR, 'data')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class_to_capture = 31
additional_dataset_size = 150

cap = cv2.VideoCapture(0)
if not os.path.exists(os.path.join(DATA_DIR, str(class_to_capture))):
    os.makedirs(os.path.join(DATA_DIR, str(class_to_capture)))

print('Ready to start. Press "Q" to begin capturing additional data for class {}'.format(class_to_capture))

done = False
while not done:
    ret, frame = cap.read(0)
    cv2.putText(frame, 'Press "Q" to start capturing!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(25)
    if key == ord('q'):
        done = True

print('Collecting additional data for class {}'.format(class_to_capture))

for counter in range(additional_dataset_size):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR, str(class_to_capture), f'additional_{counter}.jpg'), frame)

cap.release()
cv2.destroyAllWindows()
