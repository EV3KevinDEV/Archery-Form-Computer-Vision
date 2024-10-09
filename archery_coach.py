# archery_coach.py

import os  # Ensure 'os' is imported
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import logging
import pickle  # Added for loading the scaler

# Suppress TensorFlow Protobuf warnings (Optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

# Set up logging
logging.basicConfig(level=logging.INFO, filename='archery_coach.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logging.info(f"Using GPU: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        logging.warning(f"Could not set memory growth for the GPU: {e}")
else:
    logging.info("No GPU found, using CPU.")

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Updated PHASES dictionary with 'Idle' included
PHASES = {
    0: 'Idle',
    1: 'Stance',
    2: 'Nocking the Arrow',
    3: 'Hook and Grip',
    4: 'Set Position',
    5: 'Set-up',
    6: 'Draw',
    7: 'Anchor',
    8: 'Transfer to Holding',
    9: 'Aiming and Expansion',
    10: 'Release',
    11: 'Follow-Through'
}

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance(a, b):
    """
    Calculate the Euclidean distance between two points.
    """
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def extract_additional_pose_features(landmark_array):
    """
    Extract additional pose features like joint angles and distances.
    """
    additional_features = []
    
    # Example: Calculate angles at elbows and knees
    try:
        left_elbow = calculate_angle(
            landmark_array[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmark_array[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmark_array[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        right_elbow = calculate_angle(
            landmark_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmark_array[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmark_array[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        left_knee = calculate_angle(
            landmark_array[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmark_array[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmark_array[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        right_knee = calculate_angle(
            landmark_array[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmark_array[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmark_array[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        additional_features.extend([left_elbow, right_elbow, left_knee, right_knee])
    except IndexError as e:
        logging.warning(f"IndexError while calculating angles: {e}")
        additional_features.extend([0, 0, 0, 0])
    
    # Example: Calculate distances between shoulders and hips
    try:
        left_shoulder = landmark_array[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmark_array[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmark_array[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmark_array[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        shoulder_distance = calculate_distance(left_shoulder, right_shoulder)
        hip_distance = calculate_distance(left_hip, right_hip)
        
        additional_features.extend([shoulder_distance, hip_distance])
    except IndexError as e:
        logging.warning(f"IndexError while calculating distances: {e}")
        additional_features.extend([0, 0])
    
    # Add more angles and distances as needed
    
    return additional_features

def extract_pose_landmarks(results):
    """
    Extract pose landmarks and additional features from MediaPipe results.
    Returns a list of features.
    """
    landmarks = []
    additional_features = []
    if results.pose_landmarks:
        landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        # Extract additional features
        additional_features = extract_additional_pose_features(landmark_array)
    else:
        landmarks.extend([0]*99)  # Existing landmarks
        additional_features.extend([0]*6)  # Assuming 6 additional features
    
    return landmarks + additional_features

def extract_hand_landmarks(results):
    """
    Extract hand landmarks from MediaPipe results.
    Always returns 126 features (2 hands * 21 landmarks * 3 coordinates).
    Pads with zeros if fewer than two hands are detected.
    """
    landmarks = []
    if results.multi_hand_landmarks:
        # Limit to two hands
        for hand_landmarks_set in results.multi_hand_landmarks[:2]:
            for lm in hand_landmarks_set.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        # If less than two hands are detected, pad with zeros
        for _ in range(2 - len(results.multi_hand_landmarks)):
            landmarks.extend([0] * 63)  # 21 landmarks × 3 coordinates
    else:
        landmarks.extend([0] * 126)  # 2 hands × 21 landmarks × 3 coordinates
    return landmarks

class ArcheryCoachApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Archery Coach')
        self.setGeometry(100, 100, 900, 600)

        # Video Feed Label
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setGeometry(10, 10, 640, 480)

        # Feedback Text
        self.feedback_label = QtWidgets.QLabel(self)
        self.feedback_label.setGeometry(660, 10, 230, 400)
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("font-size: 14px;")

        # Progress Bar
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(660, 420, 230, 20)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        # Start Button
        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.setGeometry(10, 500, 100, 30)
        self.start_button.clicked.connect(self.start_video)

        # Stop Button
        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.setGeometry(120, 500, 100, 30)
        self.stop_button.clicked.connect(self.stop_video)

        # Record Session Button
        self.record_button = QtWidgets.QPushButton('Record Session', self)
        self.record_button.setGeometry(230, 500, 120, 30)
        self.record_button.clicked.connect(self.toggle_recording)
        self.is_recording = False

        # Timer for video feed
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize variables
        self.cap = None
        self.sequence = []
        self.sequence_length = 30  # Must match the sequence length used during training

        # Load the trained LSTM model
        self.phase_labels = [PHASES[i] for i in range(len(PHASES))]
        self.model = self.load_model()

        # Load the scaler
        self.scaler = self.load_scaler()

        # Variables for shot cycle detection and recording
        self.current_phase = None
        self.previous_phase = None
        self.shot_frames = []
        self.shot_in_progress = False
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30  # Adjust based on actual video FPS

        # Set up directory to save shot videos
        self.shot_dir = 'saved_shots'
        os.makedirs(self.shot_dir, exist_ok=True)

        # For phase detection heuristics
        self.previous_landmarks = None
        self.movement_threshold = 0.05  # Updated threshold based on training script
        self.draw_angle_threshold = 160
        self.release_threshold = 0.1

    def load_model(self):
        # Load the entire model
        model_path = 'lstm_model_full.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logging.info('Model loaded successfully.')
            print('Model loaded successfully.')
            return model
        else:
            logging.error(f'Model file not found at {model_path}.')
            print(f'Model file not found at {model_path}.')
            sys.exit(1)

    def load_scaler(self):
        # Load the scaler used during training
        scaler_path = os.path.join('dataset', 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logging.info('Scaler loaded successfully.')
            print('Scaler loaded successfully.')
            return scaler
        else:
            logging.error(f'Scaler file not found at {scaler_path}.')
            print(f'Scaler file not found at {scaler_path}.')
            return None

    def start_video(self):
        # For video file
        file_dialog = QtWidgets.QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.fps = fps
            else:
                self.fps = 30  # Default FPS

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.processed_frames = 0

            self.timer.start(int(1000 / self.fps))
            logging.info(f'Video started: {video_path}')
            print(f'Video started: {video_path}')
        else:
            logging.info('No video selected.')
            print('No video selected.')

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.sequence = []
        self.shot_frames = []
        self.shot_in_progress = False
        self.is_recording = False
        self.record_button.setText('Record Session')
        self.progress_bar.setValue(0)
        logging.info('Video stopped.')
        print('Video stopped.')

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_button.setText('Stop Recording')
            logging.info('Recording session started.')
            QtWidgets.QMessageBox.information(self, 'Recording', 'Recording session started.')
        else:
            self.record_button.setText('Record Session')
            logging.info('Recording session stopped.')
            QtWidgets.QMessageBox.information(self, 'Recording', 'Recording session stopped.')

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            logging.warning('No frame retrieved from video.')
            QtWidgets.QMessageBox.warning(self, 'End of Video', 'No more frames to process.')
            return

        # Process frame
        image, feedback = self.process_frame(frame)

        # Convert image to Qt format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

        # Update feedback
        self.feedback_label.setText(feedback)

        # Update progress bar
        self.processed_frames += 1
        progress = int((self.processed_frames / self.total_frames) * 100)
        self.progress_bar.setValue(progress)

        # Handle recording of shot sequences
        if self.is_recording and self.shot_in_progress:
            self.shot_frames.append(image.copy())

    def process_frame(self, frame):
        feedback = ''
        # Resize frame for faster processing
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get pose estimation results
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)

        # Draw landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks_set in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks_set, mp_hands.HAND_CONNECTIONS)

        # Extract features
        pose_landmarks = extract_pose_landmarks(pose_results)
        hand_landmarks = extract_hand_landmarks(hand_results)
        features = pose_landmarks + hand_landmarks  # Total: 105 + 126 = 231 features

        # Append to sequence
        self.sequence.append(features)
        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)

        # Time-Series Analysis
        if len(self.sequence) == self.sequence_length:
            sequence_input = np.array(self.sequence)  # Shape: (30, 231)

            # Apply scaling if scaler is available
            if self.scaler:
                sequence_input = self.scaler.transform(sequence_input.reshape(-1, sequence_input.shape[-1]))
                sequence_input = sequence_input.reshape(self.sequence_length, -1)

            sequence_input = np.expand_dims(sequence_input, axis=0)  # Shape: (1, 30, 231)

            try:
                predicted_phase = self.model.predict(sequence_input)
                phase_index = np.argmax(predicted_phase)
                phase_name = self.phase_labels[phase_index]
                confidence = predicted_phase[0][phase_index]

                self.previous_phase = self.current_phase
                self.current_phase = phase_name

                # Detect shot cycle start and end
                self.detect_shot_cycle()

                # Provide enhanced feedback
                feedback = self.get_enhanced_feedback(pose_results, hand_results, phase_name, frame, confidence)
            except Exception as e:
                logging.error(f'Error during model prediction: {e}')
                feedback = 'Error in prediction.'
        else:
            feedback = 'Collecting data...'

        return frame, feedback

    def detect_shot_cycle(self):
        # Detect the start and end of a shot cycle based on phase transitions
        if self.current_phase == 'Set-up' and not self.shot_in_progress:
            self.shot_in_progress = True
            self.shot_frames = []
            logging.info('Shot cycle started.')
        elif self.current_phase == 'Release' and self.shot_in_progress:
            self.shot_in_progress = False
            logging.info('Shot cycle ended.')
            if self.is_recording:
                self.save_shot_video()

    def save_shot_video(self):
        # Save the recorded shot frames as a video file
        shot_number = len(os.listdir(self.shot_dir)) + 1
        video_path = os.path.join(self.shot_dir, f'shot_{shot_number}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        for frame in self.shot_frames:
            out_video.write(frame)
        out_video.release()
        logging.info(f'Shot video saved: {video_path}')
        QtWidgets.QMessageBox.information(self, 'Shot Saved', f'Shot video saved as {video_path}')

    def get_enhanced_feedback(self, pose_results, hand_results, phase_name, frame, confidence):
        messages = [f'Current Phase: {phase_name}', f'Confidence: {confidence:.2f}']

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

            # Calculate shoulder alignment
            shoulder_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)
            if shoulder_angle < 160:
                messages.append(f'Shoulder Alignment: {shoulder_angle:.2f}° - Keep shoulders level.')
                cv2.putText(frame, 'Adjust Shoulders', (int(right_shoulder[0]*self.frame_width), int(right_shoulder[1]*self.frame_height)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                messages.append(f'Shoulder Alignment: {shoulder_angle:.2f}° - Good alignment.')

            # Calculate torso twist
            hip_angle = calculate_angle(left_hip, left_shoulder, right_shoulder)
            if hip_angle > 20:
                messages.append(f'Torso Twist: {hip_angle:.2f}° - Minimize torso rotation.')
                cv2.putText(frame, 'Reduce Torso Twist', (int(left_shoulder[0]*self.frame_width), int(left_shoulder[1]*self.frame_height)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                messages.append(f'Torso Twist: {hip_angle:.2f}° - Good posture.')

            # Analyzing elbow angle
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if elbow_angle < 160:
                messages.append(f'Elbow Angle: {elbow_angle:.2f}° - Straighten your elbow.')
                cv2.putText(frame, 'Straighten Elbow', (int(left_elbow[0]*self.frame_width), int(left_elbow[1]*self.frame_height)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                messages.append(f'Elbow Angle: {elbow_angle:.2f}° - Good job.')

        # Hand and Finger Tracking
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Check thumb and index finger distance
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = np.linalg.norm(np.array([thumb_tip.x - index_finger_tip.x,
                                                    thumb_tip.y - index_finger_tip.y,
                                                    thumb_tip.z - index_finger_tip.z]))
                if distance > 0.05:
                    messages.append('Grip: Thumb and index finger are too far apart.')
                    cv2.putText(frame, 'Adjust Grip', (int(thumb_tip.x*self.frame_width), int(thumb_tip.y*self.frame_height)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    messages.append('Grip looks good.')

        return '\n'.join(messages)

# Run the application
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ArcheryCoachApp()
    window.show()
    sys.exit(app.exec_())
