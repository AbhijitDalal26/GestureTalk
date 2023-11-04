# GestureTalk
Welcome to my GitHub repository! Here, you will find my third-year project, where I have developed a Machine Learning Algorithm capable of predicting hand signs. The project includes integration into a user-friendly web API and a dedicated Embedded System.

This project represents my journey in the field of Machine Learning and Computer Vision. Feel free to explore the code, documentation, and resources related to this project. I hope you find it informative and inspiring for your own projects and learning. If you have any questions or feedback, please don't hesitate to reach out. Thank you for visiting my repository!

# Code Explenation 

**1. Image Collection**
1. Importing Libraries:
     The code starts by importing two Python libraries, `os` and `cv2` (OpenCV).
     `os` is used for working with the file system, and `cv2` is used for computer vision tasks.
2. Setting Up Directories:
     The script defines two directory paths:
     1. `BASE_DIR`: The base directory where the Gesture Talk project is located. 
     2. `DATA_DIR`: A subdirectory within the project directory where the collected image data will be stored. This path is created by joining `BASE_DIR` and 'data'.
3. Creating Directories:
     The code checks if the `DATA_DIR` exists, and if not, it creates the directory using `os.makedirs()`.
4. Defining Constants:
    The script defines two important constants:
     1. `number_of_classes`: The number of different sign language classes or hand signs you want to collect data for (28 in this case).
     2. `dataset_size`: The number of images you want to collect for each class (150 in this case).
5. Initializing Webcam:
    The code opens a connection to the webcam using `cv2.VideoCapture(0)` .
where 0 typically represents the default camera. 

6. Collecting Data for Each Class:
   The script enters a loop that iterates through each class (from 0 to `number_of_classes - 1`).
   It checks if a directory for the current class exists in `DATA_DIR` and creates it if it doesn't.

     It then enters a loop to capture images:
     - A message is displayed on the webcam feed, asking the user to press "Q" to start capturing images.
     - The webcam feed is displayed using `cv2.imshow()` and waits for the user to press the                   "Q" key.
     - When "Q" is pressed, the loop for capturing images begins.

   - Inside the image capture loop:
     - The webcam feed is continuously captured using `cap.read()`.
     - Images are displayed in the window.
     - Images are saved in the corresponding class directory in the `DATA_DIR` with filenames like '0.jpg', '1.jpg', and so on.
     - The loop continues until `dataset_size` images are captured for the current class.

7. Releasing Resources:
    After data collection for all classes is complete, the script releases the webcam using `cap.release()` and closes all OpenCV windows with `cv2.destroyAllWindows()`.

In summary, this code sets up a data collection pipeline for an Gesture Talk project, allowing you to capture images of various sign language hand signs for multiple classes. It organizes the captured images into directories for each class in a specified data directory. 

**2. Create Dataset**
1. Imports:
     The code begins by importing the necessary libraries:
     1. `os`: For file and directory operations.
     2. `pickle`: For serializing and deserializing Python objects.
     3. `cv2`: For working with OpenCV, an open-source computer vision library.
     4. `mediapipe`: A library for building real-time multi-modal (e.g., hands, face, pose) perceptual pipelines.

2. MediaPipe Hands Initialization:
     It initializes MediaPipe Hands with specific configuration options:
     - `static_image_mode=True`: Configures MediaPipe to work with static images.
     - `min_detection_confidence=0.3`: Sets the minimum detection confidence threshold.

3. Directory Paths:
     It gets the absolute path of the script's directory using `os.path.dirname(os.path.abspath(__file__))`.
    Constructs the full file path for the 'data' directory within the script's directory.

4. Data Collection and Processing:
     The code initializes empty lists `data` and `labels` to store processed data and corresponding labels (class names).
1.	It iterates through the directories within the 'data' directory.
2.	For each image file within these directories:
It initializes empty lists `x_` and `y_` to store x and y coordinates of hand landmarks.
3.	It loads the image using OpenCV.
4.	Converts the image to RGB format since MediaPipe expects RGB images.
5.	Processes the image using MediaPipe Hands to detect hand landmarks.
6.	For each detected hand in the image:
It extracts the x and y coordinates of hand landmarks and stores them in `x_` and `y_` lists.
7.	It normalizes the coordinates by subtracting the minimum x and y values to make them relative to the hand's position within the image.
8.	Appends the normalized coordinates to the `data_aux` list.
9.	Appends the `data_aux` list to the `data` list.
10.	Appends the directory name (class label) to the `labels` list.

5. Data Serialization:
   - It serializes the collected data and labels using the `pickle` module.
   - The data is saved as a dictionary containing the 'data' and 'labels' lists to a file named 'data.pickle'.

6. Closing File:
     The 'with' block automatically closes the 'data.pickle' file.

In summary, this code loads images of hand gestures, detects hand landmarks using MediaPipe Hands, normalizes the coordinates, and saves the processed data and labels to a 'data.pickle' file for use in further machine learning or data analysis tasks.

**3. Train Model**
1. Imports:
    The code starts by importing the necessary libraries, including `pickle` for data serialization, `RandomForestClassifier` for the machine learning model, and various modules from `sklearn` for data splitting and performance evaluation.

2. Loading Data:
    The code loads the previously serialized data from the 'data.pickle' file. This data includes features (stored in `data`) and corresponding labels (stored in `labels`).

3. Data Splitting:
    The data is split into training and testing sets using the `train_test_split` function. It uses 80% of the data for training (`x_train` and `y_train`) and 20% for testing (`x_test` and `y_test`).
   - The `stratify` parameter ensures that the class distribution is maintained in both the training and testing datasets.

4. Model Initialization:
   A Random Forest classifier is initialized using `RandomForestClassifier()`.

5. Model Training:
     The model is trained on the training data using `model.fit(x_train, y_train)`.

6. Model Prediction:
    The model is used to make predictions on the test data with `model.predict(x_test)`, and the predictions are stored in `y_predict`.

7. Performance Evaluation:
    The accuracy of the model is calculated using `accuracy_score(y_predict, y_test)`. The accuracy score measures the proportion of correctly classified samples in the test dataset.

8. Printing Results:
    The code prints the accuracy score, indicating what percentage of samples were classified correctly.

9. Model Serialization:
   - The trained model is serialized using `pickle.dump()` and saved to a file named 'model.p'.

10. File Closing:
   - The file containing the serialized model is explicitly closed using `f.close()`.
This code snippet demonstrates a typical machine learning workflow, including data loading, data splitting, model training, performance evaluation, and model serialization for later use. It uses a Random Forest classifier, which is a popular ensemble learning algorithm for classification tasks.

**4. Model_1**
1. Imports:
    The code begins by importing various libraries, including OpenCV (`cv2`), MediaPipe (`mp`), NumPy (`np`), `pickle` for data serialization, `time`, and `pyttsx3` for text-to-speech functionality.
2. Initialization:
•	It initializes the webcam (camera) using `cv2.VideoCapture(0)`.
•	It initializes MediaPipe Hands, enabling hand landmark detection with specific configuration options. MediaPipe is a library that provides pre-trained models for various tasks, including hand landmarks detection.
•	loads a pre-trained machine learning model from a 'model.p' file using `pickle`.
•	Define a dictionary (`labels_dict`) that maps class labels (numbers) to corresponding hand gestures or letters. 
3. Prediction Interval:
    set a prediction interval (2 seconds) to control the frequency of making predictions.

4. Text-to-Speech Engine Initialization:
              It initializes the `pyttsx3` text-to-speech engine for voice output.

5. Create a Window:
    It creates an OpenCV window with the title "Hand Gesture Recognition."

6. Main Loop:
    The code enters the main loop for capturing and processing frames from the webcam.
    It processes each frame to detect hand landmarks using MediaPipe Hands.

7. Drawing Landmarks:
    If hand landmarks are detected, it draws landmarks and connections on the frame using `mp_drawing.draw_landmarks`.

8. Gesture Prediction:
     The code tracks the time elapsed since the last prediction and makes a prediction if the time interval (`prediction_interval`) has passed.
    For each detected hand:
      1.  It collects hand landmark data and normalizes the coordinates.
      2.  It passes the normalized data to the pre-trained model for prediction.
      3.  The predicted class is mapped to a gesture (letter) using the `labels_dict`.
      4.  The predicted gesture is printed and spoken using text-to-speech.
      5.  The previous time (`prev_time`) is updated for the timing of predictions.

9. Display Frame:
      The frame with hand landmarks and gestures is displayed in the OpenCV window.

10. Exit Loop:
      The loop can be exited by pressing the 'q' key.
11. Release Resources :
    After exiting the loop, it releases the camera and closes the OpenCV window.

This code provides real-time hand gesture recognition, detecting hand landmarks and speaking out the recognized gestures using text-to-speech.

**5. Model_2**
Explenation Comming Soon.
