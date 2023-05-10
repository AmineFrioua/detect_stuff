import cv2

# Define the red region of interest
roi_color = (0, 0, 255)  # red
roi_top_left = (100, 100)
roi_bottom_right = (300, 300)

# Create HOG descriptor object for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the video file for processing
video_path = "detect_stuff/sample.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through each frame of the video and detect humans
while cap.isOpened():
    # Read the next frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect humans in the frame
    humans, _ = hog.detectMultiScale(
        frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Check if any detected humans are in the red region of interest
    for (x, y, w, h) in humans:
        if x + w >= roi_top_left[0] and y + h >= roi_top_left[1] and x <= roi_bottom_right[0] and y <= roi_bottom_right[1]:
            # Draw a red box around the human and display an error message
            cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 2)
            cv2.putText(frame, "Error: Do not enter!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, roi_color, 2)
        else:
            # Draw a green box around the human
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the red region of interest on the frame
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, roi_color, 2)

    # Display the resulting frame
    cv2.imshow('Human Detection with Red Zone', frame)

    # Check for keypresses to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
