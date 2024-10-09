# train_lstm_model.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import pickle
import glob
from tqdm import tqdm  # For loading bar
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, filename='training.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logging.info(f"Using GPU: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        logging.warning("Could not set memory growth for the GPU.")
else:
    logging.info("No GPU found, using CPU.")

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
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

# Heuristic thresholds
movement_threshold = 0.05
draw_angle_threshold = 160
release_threshold = 0.1

# Sequence length for LSTM input
SEQUENCE_LENGTH = 30  # Adjust based on your requirements

# Directories
VIDEO_FOLDER = 'videos'  # Folder containing videos
DATASET_FOLDER = 'dataset'  # Folder to save processed data
os.makedirs(DATASET_FOLDER, exist_ok=True)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

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

def initialize_media_pipe():
    """
    Initialize MediaPipe Pose and Hands instances.
    This function will be called within each worker process.
    """
    global pose, hands
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_video(video_path):
    """
    Process a single video to extract sequences of features and labels.
    Returns:
        X: NumPy array of shape (num_sequences, SEQUENCE_LENGTH, feature_length)
        y: NumPy array of shape (num_sequences,)
    """
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    current_sequence = []
    current_phase = 'Idle'
    phase_labels = list(PHASES.values())
    previous_landmarks = None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_path)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get pose estimation results
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)

        # Extract features
        pose_landmarks = extract_pose_landmarks(pose_results)
        hand_landmarks = extract_hand_landmarks(hand_results)
        features = pose_landmarks + hand_landmarks  # Total: 105 + 126 = 231 features

        # Debugging: Check feature length
        if len(features) != 231:
            logging.error(f"Inconsistent feature length: {len(features)} in video {video_path}")
            continue  # Skip this frame

        # Convert landmarks to numpy array for calculations
        if pose_results.pose_landmarks:
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark])

            # Calculate features for phase detection
            left_shoulder = landmark_array[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmark_array[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmark_array[mp_pose.PoseLandmark.LEFT_WRIST.value]

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Movement detection
            if previous_landmarks is not None:
                movement = np.linalg.norm(landmark_array - previous_landmarks, axis=1).mean()
            else:
                movement = 0

            # Detect phases based on heuristics
            if current_phase == 'Idle' and movement < movement_threshold:
                current_phase = 'Stance'

            elif current_phase == 'Stance' and movement > movement_threshold:
                current_phase = 'Set-up'

            elif current_phase == 'Set-up' and left_elbow_angle < draw_angle_threshold:
                current_phase = 'Draw'

            elif current_phase == 'Draw' and left_elbow_angle > draw_angle_threshold:
                current_phase = 'Anchor'

            elif current_phase == 'Anchor' and movement < movement_threshold:
                current_phase = 'Aiming and Expansion'

            elif current_phase == 'Aiming and Expansion' and movement > release_threshold:
                current_phase = 'Release'

            elif current_phase == 'Release' and movement < movement_threshold:
                current_phase = 'Follow-Through'

            elif current_phase == 'Follow-Through' and movement < movement_threshold:
                current_phase = 'Idle'

            # Get phase index
            phase_index = phase_labels.index(current_phase)

            # Append features and label to current sequence
            current_sequence.append({'features': features, 'label': phase_index})

            # Update previous landmarks
            previous_landmarks = landmark_array.copy()

        else:
            # No pose landmarks detected
            current_phase = 'Idle'
            phase_index = phase_labels.index(current_phase)
            previous_landmarks = None

            # Append features and label to current sequence
            current_sequence.append({'features': features, 'label': phase_index})

        pbar.update(1)

    pbar.close()
    cap.release()

    # Organize sequences for LSTM
    X = []
    y = []

    for i in range(len(current_sequence) - SEQUENCE_LENGTH):
        seq_features = [current_sequence[j]['features'] for j in range(i, i + SEQUENCE_LENGTH)]
        seq_labels = current_sequence[i + SEQUENCE_LENGTH - 1]['label']
        X.append(seq_features)
        y.append(seq_labels)

    return np.array(X), np.array(y)

def build_model(input_shape, num_classes):
    """
    Build and compile the LSTM model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    """
    Main function to process videos, train the model, and save the trained model.
    """
    # Collect data from all videos
    X_data = []
    y_data = []

    video_files = glob.glob(os.path.join(VIDEO_FOLDER, '*.mp4')) + glob.glob(os.path.join(VIDEO_FOLDER, '*.avi'))
    logging.info(f'Found {len(video_files)} video files.')
    print(f'Found {len(video_files)} video files.')

    for video_file in video_files:
        logging.info(f'Processing video: {video_file}')
        print(f'Processing video: {video_file}')
        X, y = process_video(video_file)
        if X.size == 0 or y.size == 0:
            logging.warning(f"No valid sequences found in {video_file}. Skipping.")
            print(f"No valid sequences found in {video_file}. Skipping.")
            continue
        X_data.append(X)
        y_data.append(y)

    if not X_data:
        logging.error('No data collected. Please check your video files.')
        print('No data collected. Please check your video files.')
        return

    # Concatenate all data
    X_data = np.concatenate(X_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)
    logging.info(f'Total sequences: {X_data.shape[0]}')
    print(f'Total sequences: {X_data.shape[0]}')

    # Update feature dimension based on additional features
    feature_length = len(X_data[0][0])  # e.g., 231
    logging.info(f'Feature length per frame: {feature_length}')
    print(f'Feature length per frame: {feature_length}')

    # Save processed data (optional)
    with open(os.path.join(DATASET_FOLDER, 'processed_data.pkl'), 'wb') as f:
        pickle.dump({'X': X_data, 'y': y_data}, f)
    logging.info(f'Processed data saved to {DATASET_FOLDER}/processed_data.pkl')
    print(f'Processed data saved to {DATASET_FOLDER}/processed_data.pkl')

    # Convert labels to categorical
    num_classes = len(PHASES)
    y_data_categorical = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)

    # Build and train the model
    input_shape = (SEQUENCE_LENGTH, feature_length)  # Updated feature length
    model = build_model(input_shape, num_classes)
    logging.info('Model built successfully.')

    # Define custom callback for tqdm progress bar
    class TQDMCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs):
            super(TQDMCallback, self).__init__()
            self.tqdm = tqdm(total=epochs, desc="Training")

        def on_epoch_end(self, epoch, logs=None):
            self.tqdm.set_postfix({
                'loss': f"{logs.get('loss'):.4f}",
                'val_loss': f"{logs.get('val_loss'):.4f}",
                'accuracy': f"{logs.get('accuracy'):.4f}",
                'val_accuracy': f"{logs.get('val_accuracy'):.4f}"
            })
            self.tqdm.update(1)

        def on_train_end(self, logs=None):
            self.tqdm.close()

    # Train the model
    epochs = 100
    batch_size = 32
    validation_split = 0.2

    logging.info('Starting model training.')
    print('Starting model training.')

    model.fit(
        X_data, y_data_categorical,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[TQDMCallback(epochs)]
    )

    # Save the entire model
    model_save_path = 'lstm_model_full.h5'
    model.save(model_save_path)
    logging.info(f'Model training complete. Model saved to {model_save_path}')
    print(f'Model training complete. Model saved to {model_save_path}')

if __name__ == '__main__':
    main()
