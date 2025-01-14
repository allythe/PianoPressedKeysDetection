import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.finger_extractor import FingerExtractorOpencv, FingerExtractorMediaPipe
from src.frames_extractor import FramesExtractor
from src.hands_extractor import HandsExtractorOpencv, HandsExtractorSame
from src.keys_extraction import KeysExtractorThroughLines
from src.piano_extractor import PianoExtractor
from src.logger import logger


def get_keys_extractor(type):
    if type == "lines":
        return KeysExtractorThroughLines()


def get_hands_extractor(type):
    if type == "opencv":
        return HandsExtractorOpencv()
    elif type == "same":
        return HandsExtractorSame()


def get_fingers_extractor(type):
    if type == "opencv":
        return FingerExtractorOpencv()
    elif type == "mediapipe":
        return FingerExtractorMediaPipe()


class PressedKeysDetectionPipeline:
    def __init__(self, params):
        self.logger = logger
        self.logger.info(params)

        self.video_path = params["video_path"]
        self.plot_fingertips = params["plot_fingertips"]
        self.plot_keys = params["plot_keys"]

        # lines extraction / mapping to the real piano shape
        self.frames_extractor = FramesExtractor(params["frame_per_second"])

        self.keys_extraction_type = params["keys_extraction_type"]
        self.keys_extractor = get_keys_extractor(self.keys_extraction_type)

        #  Initialize the HandsExtractor
        self.hands_extraction_type = params["hands_extraction_type"]
        self.hands_extractor = get_hands_extractor(self.hands_extraction_type)

        self.fingers_extractor_type = params["fingers_extraction_type"]
        self.fingers_extractor = get_fingers_extractor(self.fingers_extractor_type)

        # Initialize PianoExtractor
        self.piano_extractor = PianoExtractor(
            manual=params.get("manual_region_selection", True),
            show_plots=params.get("show_plots", False),
            mse_threshold=params.get("mse_threshold", 500)
        )
        # self.pressed_keys_extractor = None

    def __extract_frames(self):
        self.logger.info("Extracting frames from video")
        self.frames = self.frames_extractor(self.video_path)  # instead of link this should be the path.
        self.logger.info(f"Extracted {len(self.frames)} frames from video {self.video_path}")

    def __extract_frame_without_hands(self):
        self.logger.info("Extracting frame without hands")
        if not self.frames:
            raise ValueError("Frames have not been extracted yet. Call __extract_frames first.")

        # Simulate frame paths (you can adjust this if you have actual paths)
        frame_paths = [f"frame_{idx}.png" for idx in range(len(self.frames))]

        # Use PianoExtractor to find the clear frame
        try:
            output_dir = "output"
            clear_frame_path, original_file_name = self.piano_extractor.extract_clear_frame(self.frames, frame_paths, output_dir)

            # Save the clear frame and its metadata
            self.frame_without_hands = {
                "frame": cv2.imread(clear_frame_path),  # Load the saved frame
                "path": clear_frame_path,
                "original_name": original_file_name
            }

            self.logger.info(f"Clear frame saved at {clear_frame_path} with original file name: {original_file_name}")

        except ValueError as e:
            self.logger.error(f"Error finding clear frame: {e}")
            self.frame_without_hands = None

    def __extract_keys(self):
        self.logger.info("Extracting keys coordinates")
        self.white_keys_coords, self.black_keys_coords, self.frames_rot_angle \
            = self.keys_extractor(self.frame_without_hands)
        self.logger.info(f"Found {len(self.white_keys_coords.keys())} white keys,"
                         f" {len(self.black_keys_coords.keys())} black keys, "
                         f"rotation angle of piano is {np.round(self.frames_rot_angle, 2)} deg")

        if self.plot_keys:
            self.__draw_keys_coords(self.white_keys_coords, self.frame_without_hands)
            self.__draw_keys_coords(self.black_keys_coords, self.frame_without_hands)

    def __rotate_one_frame(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __rotate_frames(self):
        self.logger.info(f"Rotating all frames by {np.round(self.frames_rot_angle, 2)} deg"
                         f" to make piano look parallel to the image frame")
        rotated_frames = []
        for frame in self.frames:
            rotated_frames.append(self.__rotate_one_frame(frame, self.frames_rot_angle))

        self.frames = rotated_frames

    def __plot_fingertips(self, frame, fingertips, keys_fingertips):
        vis_fingers = frame.copy()

        for tip in fingertips:
            cv2.circle(vis_fingers, tip[::-1], 10, (255, 0, 0), -1)  # Draw red circles for fingertips

        plt.title(keys_fingertips.keys())
        plt.imshow(vis_fingers)

        plt.show()

    def __draw_keys_coords(self, keys_dict, image):
        for key_name in keys_dict.keys():
            key = keys_dict[key_name]
            y_ul, x_ul, y_dr, x_dr = key.coords()
            cv2.rectangle(image, (x_ul, y_ul), (x_dr, y_dr), (0, 255, 0), 2)

        cv2.imshow("keys", image)
        cv2.waitKey(0)

    def __find_fingertips_keys(self, fingertips):
        keys_fingertips = {}

        found = np.zeros(len(fingertips))

        for i, fingertip in enumerate(fingertips):
            y, x = fingertip

            for key in self.black_keys_coords.keys():
                black_key = self.black_keys_coords[key]
                y_ul, x_ul, y_dr, x_dr = black_key.coords()

                if y_ul <= y <= y_dr and x_ul <= x <= x_dr:
                    keys_fingertips[key] = fingertip
                    found[i] = True
                    break

        for i, fingertip in enumerate(fingertips):
            if found[i]:
                continue
            y, x = fingertip

            for key in self.white_keys_coords.keys():
                white_key = self.white_keys_coords[key]
                y_ul, x_ul, y_dr, x_dr = white_key.coords()

                if y_ul <= y <= y_dr and x_ul <= x <= x_dr:
                    keys_fingertips[key] = fingertip
                    found[i] = True
                    break
        print(found, keys_fingertips, len(fingertips), len(keys_fingertips.keys()))
        return keys_fingertips

    def __extract_hands_and_fingers(self):
        """
        Extract hands from the frames using the HandsExtractor.
        """
        self.logger.info("Extracting hands masks and fingertips from all frames")
        for frame in self.frames:
            hands_mask = self.hands_extractor(frame, self.frame_without_hands)
            fingertips = self.fingers_extractor(hands_mask)

            keys_fingertips = self.__find_fingertips_keys(fingertips)
            if self.plot_fingertips:
                self.__plot_fingertips(frame, fingertips, keys_fingertips)

    def __extract_pressed_keys(self):
        self.pressed_keys = self.pressed_keys_extractor(self.fingers_coords)

    def __call__(self, *args, **kwargs):
        self.__extract_frames()
        self.__extract_frame_without_hands()
        self.__extract_keys()
        self.__rotate_frames()
        self.__extract_hands_and_fingers()
        # self.__extract_pressed_keys()


def main():
    params = {}
    params["video_path"] = "C:\\Users\\Friday\\Desktop\\IACV PROJECT\\PianoPressedKeysDetection\\videos\\video1.mp4"
    params["frame_per_second"] = 2
    params["keys_extraction_type"] = "lines"
    params["hands_extraction_type"] = "same"
    params["fingers_extraction_type"] = "mediapipe"
    params["plot_fingertips"] = True
    params["plot_keys"] = True
    pipeline = PressedKeysDetectionPipeline(params)
    pipeline()


if __name__ == "__main__":
    main()
