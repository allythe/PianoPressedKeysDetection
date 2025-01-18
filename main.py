from copy import deepcopy

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.finger_extractor import FingerExtractorOpencv, FingerExtractorMediaPipe
from src.frames_extractor import FramesExtractor
from src.hands_extractor import HandsExtractorOpencv, HandsExtractorSame
from src.keys_extraction import KeysExtractorThroughLines
from src.logger import logger
from src.mp3_creator import create_mp3
from src.frame_without_hands_extractor import FrameWithoutHandsExtractor
from src.pressed_key_extractor import PressedKeyExtractorMediaPipeZ, PressedKeyExtractorMediaPipeJoints, \
    PressedKeyExtractorClassifyImg


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


def get_pressed_key_extractor(type):
    if type == "mediapipez":
        return PressedKeyExtractorMediaPipeZ()
    elif type == "mediapipejoints":
        return PressedKeyExtractorMediaPipeJoints()
    elif type == "classify":
        return PressedKeyExtractorClassifyImg()


class PressedKeysDetectionPipeline:
    def __init__(self, params):
        self.logger = logger
        self.logger.info(params)

        self.video_path = params["video_path"]
        self.plot_fingertips = params["plot_fingertips"]
        self.plot_keys = params["plot_keys"]
        self.frame_per_second = params["frame_per_second"]

        # lines extraction / mapping to the real piano shape
        self.frames_extractor = FramesExtractor(params["frame_per_second"], params["max_number_frames"])
        self.frame_without_hands_extractor = FrameWithoutHandsExtractor()

        self.keys_extraction_type = params["keys_extraction_type"]
        self.keys_extractor = get_keys_extractor(self.keys_extraction_type)

        #  Initialize the HandsExtractor
        self.hands_extraction_type = params["hands_extraction_type"]
        self.hands_extractor = get_hands_extractor(self.hands_extraction_type)

        self.fingers_extractor_type = params["fingers_extraction_type"]
        self.fingers_extractor = get_fingers_extractor(self.fingers_extractor_type)

        self.pressed_key_extraction_type = params["pressed_key_extraction_type"]
        self.pressed_keys_extractor = get_pressed_key_extractor(self.pressed_key_extraction_type)

    def __extract_frames(self):
        self.logger.info("Extracting frames from video")
        self.frames = self.frames_extractor(self.video_path)  # instead of link this should be the path.
        self.logger.info(f"Extracted {len(self.frames)} frames from video {self.video_path}")

    def __extract_frame_without_hands(self):
        self.logger.info("Extracting frame without hands")
        self.frame_without_hands = self.frame_without_hands_extractor(self.frames)

        cv2.imshow("Frame without hands", cv2.cvtColor(self.frame_without_hands, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

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

        for fingertip in fingertips:
            x, y = fingertip.x, fingertip.y
            cv2.circle(vis_fingers, (x, y), 10, (255, 0, 0), -1)  # Draw red circles for fingertips

        cv2.imshow(f"keys", vis_fingers)
        cv2.waitKey(0)

    def __draw_keys_coords(self, keys_dict, image):
        for key_name in keys_dict.keys():
            key = keys_dict[key_name]
            y_ul, x_ul, y_dr, x_dr = key.coords()
            cv2.rectangle(image, (x_ul, y_ul), (x_dr, y_dr), (0, 255, 0), 2)

        cv2.imshow("keys", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    def __find_fingertips_keys(self, fingertips):
        keys_fingertips = {}

        found = np.zeros(len(fingertips))

        for i, fingertip in enumerate(fingertips):
            y, x = fingertip.y, fingertip.x

            for key in self.black_keys_coords.keys():
                black_key = self.black_keys_coords[key]
                y_ul, x_ul, y_dr, x_dr = black_key.coords()

                if y_ul <= y <= y_dr and x_ul <= x <= x_dr:
                    keys_fingertips[key] = fingertip
                    fingertip.key_name = key
                    found[i] = True
                    break

        for i, fingertip in enumerate(fingertips):
            if found[i]:
                continue
            y, x = fingertip.y, fingertip.x

            for key in self.white_keys_coords.keys():
                white_key = self.white_keys_coords[key]
                y_ul, x_ul, y_dr, x_dr = white_key.coords()

                if y_ul <= y <= y_dr and x_ul <= x <= x_dr:
                    keys_fingertips[key] = fingertip
                    fingertip.key_name = key
                    found[i] = True
                    break
        return keys_fingertips

    def __extract_hands_and_fingers(self):
        """
        Extract hands from the frames using the HandsExtractor.
        """
        self.logger.info("Extracting hands masks and fingertips from all frames")
        for frame in self.frames:
            hands_mask = self.hands_extractor(frame, self.frame_without_hands)
            fingertips = self.fingers_extractor(hands_mask)

            # keys_fingertips = { C_1 : Fingertip ,  A#_5: Fingertip }
            keys_fingertips = self.__find_fingertips_keys(fingertips)
            self.pressed_keys_extractor(keys_fingertips)
            # recurrent algorithm to extract pressed keys

            if self.plot_fingertips:
                self.__plot_fingertips(frame, fingertips, keys_fingertips)

    def __extract_pressed_keys(self):
        self.pressed_keys_history = self.pressed_keys_extractor.get_history(self.white_keys_coords,
                                        self.black_keys_coords, self.frames, self.frame_without_hands)

    def __create_mp3(self):
        self.logger.info("Creating mp3")
        steps = []
        for i in range(len(self.pressed_keys_history)):
            cur_step = []
            for key in self.pressed_keys_history[i]:
                if self.pressed_keys_history[i][key].press:
                    cur_step.append(key)
            steps.append(cur_step)

        print(steps)

        create_mp3(steps, "test_music.mp3", step_duration_ms=int(1000/self.frame_per_second))

    def __draw_bbox_key(self, image, key_name):
        if "#" in key_name:
            key = self.black_keys_coords[key_name]
        else:
            key = self.white_keys_coords[key_name]
        y_ul, x_ul, y_dr, x_dr = key.coords()
        cv2.rectangle(image, (x_ul, y_ul), (x_dr, y_dr), (0, 255, 0), 2)
        return image

    def __draw_video_pressed_keys(self):
        self.logger.info("Drawing pressed keys")

        for i, frame in enumerate(self.frames):
            draw_frame = deepcopy(frame)
            draw_frame = cv2.cvtColor(draw_frame, cv2.COLOR_RGB2BGR)
            history_step = self.pressed_keys_history[i]
            for key in history_step.keys():
                if history_step[key].press == True:
                    draw_frame = self.__draw_bbox_key(draw_frame, key)

            cv2.imshow("pressed keys", draw_frame)
            cv2.waitKey(0)

    def __call__(self, *args, **kwargs):
        self.__extract_frames()
        self.__extract_frame_without_hands()
        self.__extract_keys()
        self.__rotate_frames()
        self.__extract_hands_and_fingers()
        self.__extract_pressed_keys()
        self.__draw_video_pressed_keys()
        # self.__create_mp3()


def main():
    params = {}
    params["video_path"] = "videos/video3.mp4"
    params["frame_per_second"] = 20
    params["max_number_frames"] = 1000
    params["keys_extraction_type"] = "lines"
    params["hands_extraction_type"] = "same"
    params["fingers_extraction_type"] = "mediapipe"
    params["pressed_key_extraction_type"] = "classify"
    params["plot_fingertips"] = False
    params["plot_keys"] = True
    pipeline = PressedKeysDetectionPipeline(params)
    pipeline()


if __name__ == "__main__":
    main()
