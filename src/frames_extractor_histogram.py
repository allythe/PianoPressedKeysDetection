import cv2
import matplotlib.pyplot as plt
import numpy as np

def calculate_similarity(frame1, frame2):
    """
    Calculate similarity between two frames using histogram comparison.

    :param frame1: First frame.
    :param frame2: Second frame.
    :return: Similarity score (0-1 range).
    """
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

def plot_similarity(video_path, frame_per_second):
    """
    Plots the similarity of frames in a video.

    :param video_path: Path to the video file.
    :param frame_per_second: Frame extraction rate.
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate // frame_per_second)

    frame_count = 0
    prev_frame = None
    similarities = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to grayscale for similarity comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compare with the previous frame
        if prev_frame is not None:
            similarity = calculate_similarity(prev_frame, gray_frame)
            similarities.append(similarity)

        prev_frame = gray_frame
        frame_count += 1

    video_capture.release()

    # Plot the similarity values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(similarities) + 1), similarities, marker='o')
    plt.title("Frame-to-Frame Similarity")
    plt.xlabel("Frame Number")
    plt.ylabel("Similarity (0-1)")
    plt.grid(True)
    plt.show()

# Example usage
video_path = "videos/video1.mp4"
frame_per_second = 2  # Example: extract 2 frames per second
plot_similarity(video_path, frame_per_second)

