import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained object detection models
animal_model_path = "path/to/animal/model/directory"
animal_model = tf.saved_model.load(animal_model_path)

human_model_path = "path/to/human/model/directory"
human_model = cv2.CascadeClassifier(human_model_path)

# Define the categories that the animal model can detect
animal_categories = [
    "cat",
    "dog",
    "horse",
    "cow",
    "sheep",
    # add more categories as needed
]

# Open the video file for processing
video_path = "path/to/video/file"
cap = cv2.VideoCapture(video_path)

# Loop through each frame of the video and detect humans and animals
while cap.isOpened():
    # Read the next frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect animals with the TensorFlow model
    animal_boxes = []
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, 0)
    outputs = animal_model(image)
    boxes = np.array(outputs["detection_boxes"])[0]
    scores = np.array(outputs["detection_scores"])[0]
    classes = np.array(outputs["detection_classes"])[0].astype(np.int32)
    for i in range(len(boxes)):
        if scores[i] >= 0.5 and animal_categories[classes[i]-1] in animal_categories:
            animal_boxes.append(boxes[i])

    # Detect humans with the OpenCV model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    human_boxes = human_model.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    # Draw boxes around the animals in the frame
    for box in animal_boxes:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Draw boxes around the humans in the frame
    for (x, y, w, h) in human_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Human and Animal Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
