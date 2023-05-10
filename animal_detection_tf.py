import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained object detection model
model_path = "path/to/model/directory"
model = tf.saved_model.load(model_path)

# Define the categories that the model can detect
categories = [
    "cat",
    "dog",
    "horse",
    "cow",
    "sheep"
]

# Open the video file for processing
video_path = "path/to/video/file"
cap = cv2.VideoCapture(video_path)

# Loop through each frame of the video and detect animals
while cap.isOpened():
    # Read the next frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the image for input to the model
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, 0)

    # Make predictions with the model
    outputs = model(image)

    # Extract the relevant information from the predictions
    boxes = np.array(outputs["detection_boxes"])[0]
    scores = np.array(outputs["detection_scores"])[0]
    classes = np.array(outputs["detection_classes"])[0].astype(np.int32)

    # Filter the results to only include animals
    animal_boxes = []
    for i in range(len(boxes)):
        if scores[i] >= 0.5 and categories[classes[i]-1] in categories:
            animal_boxes.append(boxes[i])

    # Draw boxes around the animals in the frame
    for box in animal_boxes:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Animal Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
