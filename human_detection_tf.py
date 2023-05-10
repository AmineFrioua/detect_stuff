# This is a  setup

# pip install opencv-python

# Tenserflow install
# https://www.tensorflow.org/install/pip#macos
# tenserflow mac(arm_64) -> https://developer.apple.com/metal/tensorflow-plugin/


# I tried to make instructions more advance but it dosent work on my enviroment anymore -_-

# INSTALl
# python -m venv venv
# source venv/bin/activate
# curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
# bash Miniconda3-latest-MacOSX-x86_64.sh
# conda activate tf
# python -m pip install tensorflow-macos
# python -m pip install tensorflow-metal

# RUNNING
# source venv/bin/activate
# conda activate tf



# Tenserflow install
# https://www.tensorflow.org/install/pip#macos
# tenserflow mac(arm_64) -> https://developer.apple.com/metal/tensorflow-plugin/

# model download
# https://drive.google.com/u/1/uc?id=1IJWZKmjkcFMlvaz2gYukzFx4d6mH3py5&export=download

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import tensorflow as tf

def detect_human():
    # Use a breakpoint in the code line below to debug your script.
    model = tf.keras.models.load_model('venv/VGG_coco_SSD_512x512_iter_360000.h5')
    # Load video
    cv2.startWindowThread()
    cap = cv2.VideoCapture("HumanDetection/sample.mp4")

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # resizing for faster detection
        frame = cv2.resize(frame, (300, 300))
        # using a greyscale picture, also for faster detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')
        image /= 255.0
        image = image.reshape(1, 300, 300, 3)
        # detect people in the image
        # returns the bounding boxes for the detected objects
        output = model.predict(image)

        # Process output
        boxes = output[0, :, :4]
        classes = output[0, :, 4:]
        scores = output[0, :, 5:]

        # Select human detections
        human_boxes = []
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > 0.5:
                human_boxes.append(boxes[i])

        # Draw human boxes on image
        for box in human_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (int(xmin * image.shape[1]), int(ymin * image.shape[0])),
                          (int(xmax * image.shape[1]), int(ymax * image.shape[0])), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Preprocess image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))
        image = image.astype('float32')
        image /= 255.0
        image = image.reshape(1, 300, 300, 3)

        # Perform human detection
        output = model.predict(image)

        # Process output
        boxes = output[0, :, :4]
        classes = output[0, :, 4:]
        scores = output[0, :, 5:]

        # Select human detections
        human_boxes = []
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > 0.5:
                human_boxes.append(boxes[i])

        # Draw human boxes on image
        for box in human_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (int(xmin * image.shape[1]), int(ymin * image.shape[0])),
                      (int(xmax * image.shape[1]), int(ymax * image.shape[0])), (0, 255, 0), 2)

    # Show image
        cv2.imshow('Human detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
    cap.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detect_human()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
