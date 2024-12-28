import cv2
import numpy as np
import os
class HandsExtractor:
    def __init__(self):
        # Define the lower and upper bounds for the YCrCb color filter
        # These values help detect skin color
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

    def extract_hands_mask(self, frame, background):
        """
        Extract a binary mask of hands from the given frame with refined processing.

        Args:
            frame (numpy.ndarray): The current frame of the video.
            background (numpy.ndarray): The static background image without hands.

        Returns:
            numpy.ndarray: A refined binary mask where hands are white (255) and the rest is black (0).
        """
        # Background subtraction
        diff = cv2.absdiff(frame, background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        # Skin color filtering in YCrCb space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)

        # Combine masks
        combined_mask = cv2.bitwise_and(thresh, mask)

        # Morphological operations for noise reduction
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        return combined_mask



def process_video_frames(video_name, background_path):
    """
    Process all frames of a video and save the extracted hand masks.

    Args:
        video_name (str): Name of the video folder containing frames.
        background_path (str): Path to the background image without hands.
    """
    # Paths
    frames_dir = f"frames/{video_name}"
    output_dir = f"masks/{video_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the background image
    background = cv2.imread(background_path)
    if background is None:
        print("Error: Could not load the background image. Check the file path.")
        return

    # Initialize the HandsExtractor
    hands_extractor = HandsExtractor()

    # Process each frame in the frames directory
    for frame_name in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not load frame {frame_name}. Skipping.")
            continue

        # Extract the hand mask
        hands_mask = hands_extractor.extract_hands_mask(frame, background)

        # Save the mask to the output directory
        mask_path = os.path.join(output_dir, f"mask_{frame_name}")
        cv2.imwrite(mask_path, hands_mask)

    print(f"All masks saved to {output_dir}.")


if __name__ == "__main__":
    video_name = "video1"  # Replace with the name of your video folder
    background_path = "frames/video1/frame_7.png"  # Replace with the path to your background image
    process_video_frames(video_name, background_path)