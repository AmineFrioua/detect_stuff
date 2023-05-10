import cv2

# Define the red region of interest
roi_color = (0, 0, 255)  # red
roi_top_left = (100, 100)
roi_bottom_right = (300, 300)

# Create HOG descriptor object for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Parameters for human detection
hit_threshold = 0.6  # minimum SVM score for a detection
scale = 1.05  # scaling factor for image pyramid

# Initialize the list of active human detections
active_detections = []

# Open the video file for processing
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through each frame of the video and detect humans
while cap.isOpened():
    # Read the next frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to reduce processing time
    frame = cv2.resize(frame, (640, 360))

    # Detect humans in the frame
    humans, scores = hog.detectMultiScale(
        frame, hitThreshold=hit_threshold, winStride=(4, 4), padding=(8, 8), scale=scale)

    # Update the list of active human detections
    for i in range(len(active_detections)):
        x, y, w, h, score = active_detections[i]
        active_detections[i] = (x, y, w, h, score + 1)

    # Check if any detected humans are in the red region of interest
    for (x, y, w, h), score in zip(humans, scores):
        if score > hit_threshold and x + w >= roi_top_left[0] and y + h >= roi_top_left[1] and x <= roi_bottom_right[0] and y <= roi_bottom_right[1]:
            # Check if this human is already being tracked
            found = False
            for i in range(len(active_detections)):
                ox, oy, _, _, oscore = active_detections[i]
                if abs(x - ox) < w/2 and abs(y - oy) < h/2:
                    active_detections[i] = (x, y, w, h, score)
                    found = True
                    cv2.putText(frame, "Error: Do not enter!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, roi_color, 2)
                    break

            # If not, add it to the list of active detections
            if not found:
                active_detections.append((x, y, w, h, score))

    # Remove inactive human detections
    active_detections = [d for d in active_detections if d[4] < 5]

    # Draw bounding boxes around active human detections
    for x, y, w, h, score in active_detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw red region of interest
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, roi_color, 2)

    # Display the processed frame
    cv2.imshow("Human detection", frame)

    # Check for spacebar press to pause the video
    if cv2.waitKey(1) == ord(" "):
        while True:
            key = cv2.waitKey(0)
            if key == ord(" "):
                break
            elif key == ord("q"):
                cap.release

# Clean up resources
cap.release()
cv2.destroyAllWindows()
