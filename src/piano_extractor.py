import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class PianoExtractor:
    def __init__(self, manual=True, show_plots=False, mse_threshold=50):
        """
        Initializes the PianoExtractor.
        :param manual: If True, allows manual piano region selection.
        :param show_plots: If True, displays debugging plots.
        :param mse_threshold: Threshold for MSE to decide frame similarity.
        """
        self.manual = manual
        self.show_plots = show_plots
        self.mse_threshold = mse_threshold
        self.piano_region = None
        self.skin_cluster_counts = []

    def get_piano_region(self, frame):
        """
        Determines the piano region either manually or automatically.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        if self.manual:
            print("Selecting piano region manually...")
            return self.select_piano_region(frame)
        else:
            return self.detect_piano_region(frame)

    def select_piano_region(self, frame):
        """
        Allows the user to manually select the piano region in a frame.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        roi = cv2.selectROI("Select Piano Region", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Piano Region")
        return roi

    def calculate_mse(self, frame1, frame2):
        """
        Calculate Mean Squared Error (MSE) between two frames.
        :param frame1: First frame (grayscale).
        :param frame2: Second frame (grayscale).
        :return: Mean Squared Error value.
        """
        return np.mean((frame1.astype("float") - frame2.astype("float")) ** 2)

    def detect_skin_clusters(self, frame):
        """
        Detects skin pixel clusters using contour analysis within the selected piano region.
        :param frame: Input video frame.
        :return: Number of detected skin clusters in the piano region.
        """
        x, y, w, h = self.piano_region
        roi_frame = frame[y:y+h, x:x+w]
        ycrcb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)

        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        skin_mask = cv2.inRange(ycrcb_frame, lower_skin, upper_skin)

        # Find contours of skin regions
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cluster_count = len([c for c in contours if cv2.contourArea(c) > 500])  # Count clusters with significant size

        self.skin_cluster_counts.append(cluster_count)

        """if self.show_plots:
            plt.figure(figsize=(6, 6))
            plt.imshow(skin_mask, cmap='gray')
            plt.title(f"Detected Skin Clusters: {cluster_count}")
            plt.axis("off")
            plt.show()"""

        return cluster_count

    def extract_clear_frame(self, frames):
        """
        Extract the first frame where no hands are present on the piano.
        :param frames: List of frames (numpy arrays).
        :return: Index of the first clear frame.
        """
        if self.piano_region is None:
            self.piano_region = self.get_piano_region(frames[0])
            print(f"Selected piano region: {self.piano_region}")

        previous_frame = None
        mse_values = []
        clear_frame_idx = None

        for idx, frame in enumerate(frames):
            x, y, w, h = self.piano_region
            cropped_gray_frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

            # Calculate skin clusters in the selected region
            skin_clusters = self.detect_skin_clusters(frame)

            if previous_frame is not None:
                mse = self.calculate_mse(previous_frame, cropped_gray_frame)
                mse_values.append(mse)

                # Condition to identify a clear frame (low MSE + minimal skin clusters)
                if mse < self.mse_threshold and skin_clusters == 0:
                    clear_frame_idx = idx
                    break

            previous_frame = cropped_gray_frame

        if self.show_plots:
            self.plot_metrics(mse_values)

        if clear_frame_idx is None:
            raise ValueError("No clear piano frame found.")
        
        return clear_frame_idx

    def plot_metrics(self, mse_values):
        """
        Plots the MSE values and skin cluster counts across all frames.
        :param mse_values: List of computed MSE values.
        """
        plt.figure(figsize=(12, 6))

        # Plot MSE values
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o', label='MSE Values')
        plt.axhline(y=self.mse_threshold, color='r', linestyle='--', label=f'Threshold ({self.mse_threshold})')
        plt.xlabel('Frame Index')
        plt.ylabel('MSE Value')
        plt.title('MSE of Cropped Regions')
        plt.legend()
        plt.grid(True)

        # Plot skin cluster counts
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.skin_cluster_counts)), self.skin_cluster_counts, marker='o', color='orange', label='Skin Clusters')
        plt.axhline(y=0, color='r', linestyle='--', label='Expected Clear Frame')
        plt.xlabel('Frame Index')
        plt.ylabel('Skin Cluster Count')
        plt.title('Skin Clusters in Selected Region')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    # Load frames from a directory into a list
    frames_dir = "C:\\Users\\Friday\\Desktop\\IACV PROJECT\\PianoPressedKeysDetection\\frames2\\video1\\cleaned"
    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]

    # Initialize the PianoExtractor
    extractor = PianoExtractor(manual=True, show_plots=True, mse_threshold=50)

    # Extract a clear piano frame from the frames list
    try:
        clear_frame_idx = extractor.extract_clear_frame(frames)
        print(f"Clear piano frame found at index: {clear_frame_idx}")

        # Save the clear frame for verification
        cv2.imwrite(f"clear_piano_frame_{clear_frame_idx}.png", frames[clear_frame_idx])
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
