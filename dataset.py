import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define dataset directory and output file
DATA_DIR = './data'
OUTPUT_FILE = 'data.pickle'

data = []   # To store the processed landmark features
labels = [] # To store corresponding class labels

# Ensure the data directory exists and is not empty
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    raise FileNotFoundError("❌ No data found in './data'. Run calculator.py to capture images first.")

# Process images from each class folder (each folder name should be the label, e.g., "0", "1", "2")
for label in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue  # Skip non-directory files
    print(f"🔍 Processing images for class {label}...")
    
    for img_name in sorted(os.listdir(class_dir)):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Warning: Could not read image {img_path}, skipping...")
            continue
        
        # Convert image to RGB and resize for consistency
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256))
        
        # Process the image with MediaPipe to get hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates from each landmark
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                
                if not x_coords or not y_coords:
                    print(f"⚠️ Warning: No valid landmarks in {img_path}, skipping.")
                    continue
                
                # Normalize landmarks by subtracting the minimum coordinate values
                min_x = min(x_coords)
                min_y = min(y_coords)
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)
                
                data.append(data_aux)
                labels.append(int(label))
        else:
            print(f"⚠️ Warning: No hand detected in {img_path}, skipping.")

# Save the processed data if any valid samples were found
if data:
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    print(f"✅ Dataset saved to {OUTPUT_FILE} with {len(data)} samples.")
else:
    print("❌ Error: No valid hand landmarks found. Check your images and try recapturing them.")

print("🎉 Processing complete!")
