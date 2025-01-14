import os

import cv2
import mediapipe as mp
import numpy as np

from src.logger import logger


class PressedKeyExtractorBase:
    def __init__(self):
        self.logger = logger
        self.logger.info("Pressed Key Extractor created")

class PressedKeyExtractorMediaPipeJoints(PressedKeyExtractorBase):
    def __init__(self):
        super().__init__()
        self.coords_history = []
        self.max_dists = {}
        self.dist_ratio = 0.5

    def get_history(self):
        for i in range(len(self.coords_history)):
            for key in self.coords_history[i].keys():
                name = f"{self.coords_history[i][key].idx}_{self.coords_history[i][key].hand}"
                max_dist = self.max_dists[name]
                cur_dist = self.coords_history[i][key].dist
                if cur_dist < max_dist * self.dist_ratio:
                    self.coords_history[i][key].set_press()

        return self.coords_history
    def __call__(self, keys_fingertips):
        # если отношение между расстоянием крайней и средней точкой
        # стало меньше чем максимальное расстояние

        idx_coord_dict = {}
        for key in keys_fingertips.keys():
            idx_coord_dict[f"{keys_fingertips[key].idx}_{keys_fingertips[key].hand}"] \
                = keys_fingertips[key]

        for key in idx_coord_dict.keys():
            if (not key in self.max_dists or
                    idx_coord_dict[key].dist > self.max_dists[key]):
                self.max_dists[key] = idx_coord_dict[key].dist

        self.coords_history.append(keys_fingertips)













class PressedKeyExtractorMediaPipeZ(PressedKeyExtractorBase):
    def __init__(self):
        super().__init__()
        self.coords_history = []
        self.analyzed_keys = []

    def get_history(self):
        return self.coords_history
    def __call__(self, keys_fingertips):
        # если z координата становится больше
        # то значит клавиша нажата

        # вот z становится больше больше больше
        # а в следующий момент меньше - z1
        # взять все предыдущие кадры где z была больше z1 и
        # сказать что это нажатая клавиша
        print(f'current step is {len(self.coords_history)}')
        analyzed_this_step = []
        if len(self.coords_history) == 0:
            self.coords_history.append(keys_fingertips)
        else:
            history_prev = self.coords_history[-1]

            for key in keys_fingertips.keys():
                cur_key = keys_fingertips[key]

                if key in history_prev.keys():
                    cur_key_past = history_prev[key]
                    diff = cur_key.z - cur_key_past.z
                    if diff > 0:
                        analyzed_this_step.append(key)

                        if not key in self.analyzed_keys:
                            self.analyzed_keys.append(key)
                    else:
                        if key in self.analyzed_keys:
                            # the key was pressed!
                            self.analyzed_keys.remove(key)

                            # find all history steps where
                            # z is between this z and prev z

                            max_z = self.coords_history[-1][key].z
                            min_z = cur_key.z

                            for i in range(len(self.coords_history)-1, -1, -1):
                                if (key in self.coords_history[i] and
                                        min_z <= self.coords_history[i][key].z <= max_z):
                                    self.coords_history[i][key].set_press()
                                    print(f'key {key} was pressed on step {i}')
                                else:
                                    break
                    print(key, cur_key.z, cur_key_past.z, diff)
                else:
                    print(key, cur_key.z)

            for key in self.analyzed_keys:
                if not key in analyzed_this_step:
                    self.analyzed_keys.remove(key)

            self.coords_history.append(keys_fingertips)
