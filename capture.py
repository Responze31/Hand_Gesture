import os
import cv2

# Directory setup
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# Parameters
number_of_classes = 3  # Number of gesture classes
dataset_size = 100     # Number of images per class
IMG_SIZE = 256         # Standardized image size (width and height)

# Function to initialize camera
def initialize_camera():
    for i in range(3):  # Try camera indices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera initialized at index {i}")
            return cap
    print("❌ Error: Could not access the camera. Check permissions!")
    return None

# Initialize camera
cap = initialize_camera()
if cap is None:
    exit()

# Data collection loop for each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)
    print(f"📸 Collecting data for class {j}")

    # Wait for user confirmation to start capturing images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read from camera")
            break
        cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Capture a set number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read from camera")
            break
        # Resize image for consistency
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Data collection complete!")
