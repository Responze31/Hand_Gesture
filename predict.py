import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from model.p
if not os.path.exists("model.p"):
    raise FileNotFoundError("❌ Error: 'model.p' not found. Train and save the model first.")
model_dict = pickle.load(open("model.p", "rb"))
if "model" not in model_dict:
    raise KeyError("❌ Error: 'model' key missing in model.p.")
model = model_dict["model"]

# Initialize camera with multiple index attempts
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera initialized at index {i}")
        break
if cap is None or not cap.isOpened():
    raise RuntimeError("❌ Error: Could not access the camera. Check permissions!")

# Setup MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Update the label mapping with your desired gesture names
labels_dict = {
    0: "Fite me",
    1: "Peace",
    2: "Good Stuff"
}

while True:
    data_aux = []
    x_coords = []
    y_coords = []
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read from camera")
        break
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Draw landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        # Process landmarks for prediction
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
            if not x_coords or not y_coords:
                print("⚠️ Warning: No hand detected, skipping frame.")
                continue
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_coords))
                data_aux.append(landmark.y - min(y_coords))
        
        # Calculate bounding box for visualization
        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10
        x2 = int(max(x_coords) * W) - 10
        y2 = int(max(y_coords) * H) - 10
        
        # Predict using the trained model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = int(prediction[0])
        predicted_character = labels_dict.get(predicted_label, "Unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    cv2.imshow("frame", frame)
    # Press 'q' to exit the prediction loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program exited successfully.")
