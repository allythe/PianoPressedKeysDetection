import numpy as np

class WhiteKeyPressExtractor:
    def __init__(self, intensity_threshold=30, debug=False):
        """
        Initialize the White Key Press Extractor.
        :param intensity_threshold: Threshold for intensity change to detect a press.
        :param debug: If True, show debug visualizations.
        """
        self.intensity_threshold = intensity_threshold
        self.debug = debug

    def is_key_pressed(self, key, previous_frame, current_frame):
        """
        Determines if a white key is pressed based on changes in intensity.
        :param key: WhiteKey object representing the key region.
        :param previous_frame: Previous frame (grayscale).
        :param current_frame: Current frame (grayscale).
        :return: True if the key is pressed, False otherwise.
        """
        y_ul, x_ul, y_dr, x_dr = key.coords()

        # Extract key region
        key_region_prev = previous_frame[y_ul:y_dr, x_ul:x_dr]
        key_region_curr = current_frame[y_ul:y_dr, x_ul:x_dr]

        # Analyze the left and right edges
        left_edge_prev = key_region_prev[:, :5]
        left_edge_curr = key_region_curr[:, :5]

        right_edge_prev = key_region_prev[:, -5:]
        right_edge_curr = key_region_curr[:, -5:]

        # Compute intensity differences
        left_diff = np.mean(left_edge_curr) - np.mean(left_edge_prev)
        right_diff = np.mean(right_edge_curr) - np.mean(right_edge_prev)

        # Determine if the key is pressed
        if left_diff > self.intensity_threshold or right_diff > self.intensity_threshold:
            if self.debug:
                print(f"Key pressed: {key}")
            return True

        return False

    def detect_pressed_keys(self, white_keys, previous_frame, current_frame):
        """
        Detects pressed keys in the current frame compared to the previous frame.
        :param white_keys: Dictionary of WhiteKey objects.
        :param previous_frame: Previous frame (grayscale).
        :param current_frame: Current frame (grayscale).
        :return: List of pressed keys.
        """
        pressed_keys = []
        for key_name, key in white_keys.items():
            if self.is_key_pressed(key, previous_frame, current_frame):
                pressed_keys.append(key_name)
        return pressed_keys

def main():
    params = {
        "video_path": "C:\\Users\\Friday\\Desktop\\IACV PROJECT\\PianoPressedKeysDetection\\videos\\video1.mp4",
        "frame_per_second": 2,
        "keys_extraction_type": "lines",
        "hands_extraction_type": "same",
        "fingers_extraction_type": "mediapipe",
        "plot_fingertips": True,
        "plot_keys": True,
        "debug": True
    }
    pipeline = PressedKeysDetectionPipeline(params)
    pipeline()

    # Print pressed keys
    for frame_idx, pressed_keys in enumerate(pipeline.pressed_white_keys):
        print(f"Frame {frame_idx + 1}: Pressed keys - {pressed_keys}")


if __name__ == "__main__":
    main()
