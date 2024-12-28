from frames_extractor import FramesExtractor
from hands_extractor import HandsExtractor
from src.keys_extraction import KeysExtractorThroughLines


def get_keys_extractor(type):
    if type == "lines":
        return KeysExtractorThroughLines


class PressedKeysDetectionPipeline():
    def __init__(self, params):
        self.video_link = params["video_link"]

        # 2d - 3d - it will be needed
        # if we will want to apply extraction of 3d coordinates
        self.video_type = params["video_type"]

        # lines extraction / mapping to the real piano shape
        self.frames_extractor = FramesExtractor(params["frame_per_second"])

        self.keys_extraction_type = params["keys_extraction_type"]
        self.keys_extractor = get_keys_extractor(self.keys_extraction_type)

        # Initialize the HandsExtractor
        self.hands_extractor = HandsExtractor()
        # Static background frame to be subtracted
        self.background_frame = params["background_frame"]

        self.fingers_extractor = None
        self.pressed_keys_extractor = None

    def __extract_frames(self):
        self.frames = self.frames_extractor(self.video_link) # instead of link this should be the path.

    def __extract_keys(self):
        self.keys_coords = self.keys_extractor(self.frames)

    def __extract_hands(self):
        """
        Extract hands from the frames using the HandsExtractor.
        """
        self.hands_coords = []
        for frame in self.frames:
            hands_mask = self.hands_extractor.extract_hands_mask(frame, self.background_frame)
            self.hands_coords.append(hands_mask)

    def __extract_fingers(self):
        self.fingers_coords = self.fingers_extractor(self.hands_coords)

    def __extract_pressed_keys(self):
        self.pressed_keys = self.pressed_keys_extractor(self.fingers_coords)

    def __call__(self, *args, **kwargs):
        self.__extract_frames()
        self.__extract_keys()
        self.__extract_hands()
        self.__extract_fingers()
        self.__extract_pressed_keys()
