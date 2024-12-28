import cv2
import numpy as np
import matplotlib.pyplot as plt

class HandsExtractor:
    def __init__(self):
        # Define the lower and upper bounds for the YCrCb color filter
        # These values help detect skin color
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    
    def extract_hands_mask(self, frame, background):
        """
        Extract a binary mask of hands from the given frame.

        Args:
            frame (numpy.ndarray): The current frame of the video.
            background (numpy.ndarray): The static background image without hands.

        Returns:
            numpy.ndarray: A binary mask where hands are white (255) and the rest is black (0).
        """
        # 1. Background subtraction to isolate moving objects (hands)
        diff = cv2.absdiff(frame, background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        # 2. Apply a YCrCb color filter to isolate skin tones
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)

        # 3. Combine the background subtraction mask and the color filter mask
        combined_mask = cv2.bitwise_and(thresh, mask)

        # 4. Apply morphological operations (erosion and dilation) to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        return combined_mask


def main():
    """
    Main function to test the HandsExtractor class.
    """
    # Input: Path to the background image and video frame
    background_path = "frames/video1/frame_7.png"  # Path to a static background image (without hands)
    frame_path = "frames/video1/frame_141.png"  # Path to a frame where hands are present
    

# Read the background and frame images
    background = cv2.imread(background_path)
    frame = cv2.imread(frame_path)

    if background is None or frame is None:
        print("Error: Could not load the images. Check the file paths.")
        return

    # Create an instance of HandsExtractor
    hands_extractor = HandsExtractor()

    # Call the extract_hands_mask method to get the binary mask
    hands_mask = hands_extractor.extract_hands_mask(frame, background)

    # Display the binary mask using Matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(hands_mask, cmap='gray')
    plt.title("Binary Mask of Hands")
    plt.axis("off")
    plt.show()

    # Optionally, save the mask to a file
    cv2.imwrite("hands_mask.png", hands_mask)
    print("Hand mask saved as 'hands_mask.png'.")

if __name__ == "__main__":
    main()