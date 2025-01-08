from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.finger_extractor import FingerExtractor
from src.frames_extractor import FramesExtractor
from src.hands_extractor import HandsExtractor
from src.keys_extraction import KeysExtractorThroughLines
from src.logger import logger


def get_keys_extractor(type):
    if type == "lines":
        return KeysExtractorThroughLines()


class PressedKeysDetectionPipeline():
    def __init__(self, params):
        self.logger = logger

        self.video_path = params["video_path"]

        # # 2d - 3d - it will be needed
        # # if we will want to apply extraction of 3d coordinates
        # self.video_type = params["video_type"]

        # lines extraction / mapping to the real piano shape
        self.frames_extractor = FramesExtractor(params["frame_per_second"])

        self.keys_extraction_type = params["keys_extraction_type"]
        self.keys_extractor = get_keys_extractor(self.keys_extraction_type)

        # # Initialize the HandsExtractor
        self.hands_extractor = HandsExtractor()

        self.fingers_extractor = FingerExtractor()
        # self.pressed_keys_extractor = None

    def __extract_frames(self):
        self.logger.info("Extracting frames from video")
        self.frames = self.frames_extractor(self.video_path)  # instead of link this should be the path.
        self.logger.info(f"Extracted {len(self.frames)} frames from video {self.video_path}")

    def __extract_frame_without_hands(self):
        # TODO: Refika
        self.frame_without_hands = self.frames[2]

        # plt.imshow(self.frame_without_hands)
        # plt.show()

    def __extract_keys(self):
        self.logger.info("Extracting keys coordinates")
        self.white_keys_coords, self.black_keys_coords, self.frames_rot_angle\
            = self.keys_extractor(self.frame_without_hands)
        self.logger.info(f"Found {len(self.white_keys_coords.keys())} white keys,"
                         f" {len(self.black_keys_coords.keys())} black keys, "
                         f"rotation angle of piano is {np.round(self.frames_rot_angle,2)} deg")

    def __rotate_one_frame(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __rotate_frames(self):
        self.logger.info(f"Rotating all frames by {np.round(self.frames_rot_angle,2)} deg"
                         f" to make piano look parallel to the image frame")
        rotated_frames = []
        for frame in self.frames:
            rotated_frames.append(self.__rotate_one_frame(frame, self.frames_rot_angle))

        self.frames = rotated_frames

    def __extract_hands_and_fingers(self):
        """
        Extract hands from the frames using the HandsExtractor.
        """

        # TODO: try to find model which can extract fingers
        self.logger.info("Extracting hands masks and fingertips from all frames")
        hands_masks = []
        for frame in self.frames:
            hands_mask = self.hands_extractor.extract_hands_mask(frame, self.frame_without_hands)
            fingertips = self.fingers_extractor.extract_fingers(hands_mask)

            vis_fingers = frame
            for tip in fingertips:
                cv2.circle(vis_fingers, tip, 10, (255, 255, 0), -1)  # Draw red circles for fingertips

            plt.subplot(311)
            plt.imshow(frame)

            plt.subplot(312)
            plt.imshow(hands_mask)

            plt.subplot(313)
            plt.imshow(vis_fingers)

            plt.show()

            hands_masks.append(hands_mask)

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
    params["video_path"] = "videos/video3.mp4"
    params["frame_per_second"] = 2
    params["keys_extraction_type"] = "lines"
    pipeline = PressedKeysDetectionPipeline(params)
    pipeline()


if __name__ == "__main__":
    main()
