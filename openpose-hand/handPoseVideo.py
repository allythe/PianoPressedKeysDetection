import cv2
import time
import numpy as np
import os

# Path to the pre-trained model and configuration files
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"

# Number of key points to detect in the hand
nPoints = 22

# Define the connections between the detected keypoints
POSE_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]

# Threshold for considering detected keypoints
threshold = 0.2

# Load input video file
video_file = "video1.mp4"
cap = cv2.VideoCapture(video_file)
hasFrame, frame = cap.read()

# Get video frame dimensions
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# Calculate aspect ratio for resizing input
aspect_ratio = frameWidth / frameHeight
inHeight = 368
inWidth = int(((aspect_ratio * inHeight) * 8) // 8)

# Define output video file writer (MP4 format for better compatibility)
vid_writer = cv2.VideoWriter('output_piano.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame.shape[1], frame.shape[0]))

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load the pre-trained neural network model
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Frame processing loop
k = 0
while True:
    k += 1
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    frameCopy = np.copy(frame)

    # Prepare input image blob for deep learning model
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)

    # Forward pass through the network to get predictions
    output = net.forward()
    print("Forward pass time: {:.2f} sec".format(time.time() - t))

    # Store detected keypoints
    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find the most likely position of the keypoint
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    # Draw skeleton by connecting detected keypoints
    for pair in POSE_PAIRS:
        partA, partB = pair
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    print("Total frame processing time: {:.2f} sec".format(time.time() - t))

    # Save the processed frame as a PNG image
    cv2.imwrite(f'results/frame_{k:04d}.png', frame)

    # Display the output frame and write to video
    cv2.imshow('Output-Skeleton', frame)
    vid_writer.write(frame)

    # Press 'ESC' key to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release video writer and resources
vid_writer.release()
cap.release()
cv2.destroyAllWindows()
