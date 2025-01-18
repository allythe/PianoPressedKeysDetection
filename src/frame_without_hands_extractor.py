import cv2
import numpy as np
import os


class FrameWithoutHandsExtractor:
    def __init__(self, manual=True, show_plots=False, mse_threshold=500):
        """
        Initializes the PianoExtractor.
        :param manual: If True, allows manual piano region selection. If False, uses automatic detection.
        :param show_plots: If True, displays debugging plots.
        :param mse_threshold: Threshold for MSE to decide if a frame is valid.
        """
        self.manual = manual
        self.show_plots = show_plots
        self.mse_threshold = mse_threshold
        self.piano_region = None

    def select_piano_region(self, frame):
        """
        Allows the user to manually select the piano region in a frame.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        roi = cv2.selectROI("Select Piano Region", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Piano Region")
        return roi

    def detect_piano_region(self, frame):
        """
        Automatically detects the piano region using edge detection and contour analysis.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour (assuming it's the piano)
        piano_region = None
        max_area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > max_area:
                max_area = area
                piano_region = (x, y, w, h)

        if piano_region is None:
            raise ValueError("Piano region could not be detected.")
        return piano_region

    def get_piano_region(self, frame):
        """
        Determines the piano region either manually or automatically.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        if self.manual:
            return self.select_piano_region(frame)
        else:
            return self.detect_piano_region(frame)

    def calculate_mse(self, frame1, frame2):
        """
        Calculate Mean Squared Error (MSE) between two frames.
        :param frame1: First frame (grayscale).
        :param frame2: Second frame (grayscale).
        :return: Mean Squared Error value.
        """
        return np.mean((frame1 - frame2) ** 2)

    def __call__(self, frames, output_dir="output"):
        """
        Extracts the first frame where the piano is clear (no hand present) from a list of frames.
        :param frames: List of frames (numpy arrays).
        :param output_dir: Directory to save the extracted frame.
        :return: Path to the saved frame and its index.
        """
        if not frames:
            raise ValueError("The frames list is empty.")

        # Get the piano region from the first frame
        self.piano_region = self.get_piano_region(frames[0])
        x, y, w, h = self.piano_region

        # Initialize variables for MSE calculation
        previous_frame = None
        for idx, frame in enumerate(frames):
            piano_frame = frame[y:y + h, x:x + w]
            gray_frame = cv2.cvtColor(piano_frame, cv2.COLOR_BGR2GRAY)

            if previous_frame is not None:
                mse = self.calculate_mse(previous_frame, gray_frame)
                if mse < self.mse_threshold:
                    # # Save the clear frame
                    # os.makedirs(output_dir, exist_ok=True)
                    # output_path = os.path.join(output_dir, f"clear_piano_frame_{idx}.png")
                    # cv2.imwrite(output_path, frame)

                    return frame

            previous_frame = gray_frame

        raise ValueError("No clear piano frame found.")


def main():
    # Load frames from a directory into a list
    frames_dir = "C:\\Users\\Friday\\Desktop\\IACV PROJECT\\PianoPressedKeysDetection\\frames\\video1\\cleaned"
    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]

    # Initialize the PianoExtractor
    extractor = FrameWithoutHandsExtractor(manual=False, show_plots=True, mse_threshold=500)

    # Extract a clear piano frame from the frames list
    try:
        clear_frame_path = extractor.extract_clear_frame(frames)
        print(f"Clear piano frame saved at: {clear_frame_path}")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
