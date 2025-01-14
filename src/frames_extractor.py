import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from src.logger import logger


class FramesExtractor:
    def __init__(self, frame_per_second, max_number_frames, deviation_threshold=1.5, show_plots=False, save_frames=False):
        """
        Initializes the FramesExtractor.

        :param frame_per_second: Number of frames to extract per second.
        :param deviation_threshold: Threshold for standard deviation to filter frames.
        """
        self.save_frames = save_frames
        self.show_plots = show_plots
        self.frame_per_second = frame_per_second
        self.deviation_threshold = deviation_threshold
        self.max_number_frames = max_number_frames
        self.logger = logger
        self.logger.info("Frames Extractor created")

    def _calculate_similarity(self, frame1, frame2):
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

    def __call__(self, video_path):
        """
        Extracts frames based on overall similarity deviations and saves them as PNG files.

        :param video_path: Path to the video file.
        :return: List of file paths of the extracted frames.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create the frames directory based on the video name
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if self.save_frames:
            output_dir = os.path.join("frames", video_name)
            os.makedirs(output_dir, exist_ok=True)

        # Open the video file
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate // self.frame_per_second)

        similarities = []
        frames = []
        selected_frames = []
        first_frame = None
        frame_count = 0
        appended_frames = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Process only selected frames based on frame_interval
            if frame_count % frame_interval == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
                selected_frames.append(gray_frame)
                appended_frames +=1

                if first_frame is None:
                    first_frame = gray_frame
                else:
                    similarity = self._calculate_similarity(first_frame, gray_frame)
                    similarities.append(similarity)

            frame_count += 1
            if appended_frames>self.max_number_frames:
                break

        video_capture.release()

        # Calculate mean and standard deviation of all similarities
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        # Filter frames in the middle range (close to the mean similarity)
        saved_frames = []
        for i, similarity in enumerate(similarities):
            if (mean_similarity - self.deviation_threshold * std_similarity <= similarity
                    <= mean_similarity + self.deviation_threshold * std_similarity):

                if self.save_frames:
                    frame_filename = os.path.join(output_dir, f"frame_{i}.png")
                    cv2.imwrite(frame_filename, frames[i])
                saved_frames.append(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))

        if self.show_plots:
            # Plot similarity scores
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(similarities)), similarities, marker='o', label='Similarity Scores')
            plt.axhline(mean_similarity, color='green', linestyle='--', label='Mean Similarity')
            plt.axhline(mean_similarity + self.deviation_threshold * std_similarity, color='red', linestyle='--',
                        label='Upper Threshold')
            plt.axhline(mean_similarity - self.deviation_threshold * std_similarity, color='red', linestyle='--',
                        label='Lower Threshold')
            plt.title("Frame Similarity Analysis - Middle Range Extraction")
            plt.xlabel("Frame Index")
            plt.ylabel("Similarity")
            plt.legend()
            plt.grid(True)
            plt.show()

        return saved_frames


def main():
    """
    Test the FramesExtractor class here.
    """
    video_path = "../videos/video1.mp4"
    frame_per_second = 25  # Example: extract 2 frames per second
    deviation_threshold = 1  # Threshold for standard deviation filtering

    frames_extractor = FramesExtractor(frame_per_second, deviation_threshold)
    extracted_frames = frames_extractor(video_path)

    # TODO: Alisa derive optimal frame_per_second value

    print(f"Extracted {len(extracted_frames)} frames. Frames saved in 'frames' folder.")


if __name__ == "__main__":
    main()
