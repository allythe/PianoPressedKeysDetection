import os

import cv2
import mediapipe as mp
import numpy as np

from src.logger import logger


class PressedKeyExtractorBase:
    def __init__(self):
        self.logger = logger
        self.logger.info("Pressed Key Extractor created")


class PressedKeyExtractorMediaPipeZ(PressedKeyExtractorBase):
    def __init__(self):
        super().__init__()

    def __call__(self, keys_fingertips):
        pass

