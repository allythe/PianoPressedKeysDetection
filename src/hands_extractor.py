import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from src.logger import logger

class HandsExtractorBase:
    def __init__(self):
        self.logger = logger
        self.logger.info("Hands Extractor created")

class HandsExtractorOpencv(HandsExtractorBase):

    def __init__(self, save_debug = False, show_plot = True):
        super.__init__()
        # Define the lower and upper bounds for the YCrCb color filter
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

        self.save_debug = save_debug
        self.show_plot = show_plot

        """
        values on paper 
        self.lower_ycrcb = np.array([70, 141, 0], dtype=np.uint8)  # Y_MIN, Cr_MIN, Cb_MIN
        self.upper_ycrcb = np.array([198, 255, 256], dtype=np.uint8)  # Y_MAX, Cr_MAX, Cb_MAX
        """

    def __call__(self, frame, background):
        # 1. Arka plan çıkarımı
        diff = cv2.absdiff(frame, background)

        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)  # Gürültüyü azalt
        _, thresh = cv2.threshold(diff_gray, 50, 255, cv2.THRESH_BINARY)  # Daha yüksek eşik

        # 2. YCrCb renk filtresi
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)

        # 3. Maskeleri birleştir
        combined_mask = cv2.bitwise_and(thresh, mask)

        # 4. Morphological işlemler
        kernel = np.ones((5, 5), np.uint8)  # Kernel boyutunu artır
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 5. Kontur bazlı filtreleme
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(combined_mask)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Daha düşük eşik
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if self.save_debug:
            cv2.imwrite("debug_diff_gray.png", diff_gray)  # Arka plan farkı
            cv2.imwrite("debug_thresh.png", thresh)  # Threshold sonrası
            cv2.imwrite("debug_mask.png", mask)  # YCrCb filtresi
            cv2.imwrite("debug_combined.png", combined_mask)  # Kombine maske
            cv2.imwrite("debug_filtered.png", filtered_mask)  # Kontur sonrası maske

        return filtered_mask

class HandsExtractorSame(HandsExtractorBase):
    def __init__(self):
        super().__init__()

    def __call__(self, image, _):
        return image

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
    hands_extractor = HandsExtractorOpencv()

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
        # cv2.imwrite(mask_path, hands_mask)

    print(f"All masks saved to {output_dir}.")



if __name__ == "__main__":
    # video_name = "video1"  # Replace with the name of your video folder
    # background_path = "frames/video1/frame_7.png"  # Replace with the path to your background image
    # process_video_frames(video_name, background_path)

    test_mediapipe()
