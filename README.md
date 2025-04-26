# Hand Gesture Recognition

A computer vision project that recognizes hand gestures in real-time using MediaPipe and machine learning.

## Overview

This project captures hand gestures via webcam, processes them using MediaPipe's hand tracking, and classifies them using a Random Forest model.

## Files

- `calculator.py` - Captures training images for different gesture classes
- `create_dataset.py` - Processes images to extract hand landmarks and create training data
- `train.py` - Trains a Random Forest classifier on the processed data
- `inference.py` - Real-time hand gesture recognition using webcam

## Setup

1. Install dependencies:
```
pip install opencv-python mediapipe numpy scikit-learn
```

2. Capture training data:
```
python calculator.py
```

3. Process the captured images:
```
python create_dataset.py
```

4. Train the model:
```
python train.py
```

5. Run the recognition system:
```
python inference.py
```

## Usage

- Run `calculator.py` to capture training images for each gesture class
- Collected images will be saved in the `./data` directory
- Process images with `create_dataset.py` to create `data.pickle`
- Train the model with `train.py` to create `model.p`
- Run `inference.py` for real-time gesture recognition
- Press 'q' to exit the recognition program

## Customization

Edit the `labels_dict` in `inference.py` to change the gesture names:
```python
labels_dict = {
    0: "Fite me",
    1: "Peace",
    2: "Good Stuff"
}
```
