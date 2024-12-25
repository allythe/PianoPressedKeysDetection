import cv2
import os

class FramesExtractor:
    def __init__(self, frame_per_second):
        self.frame_per_second = frame_per_second

    def __call__(self, video_path):
        """
        Extracts frames from the video at the specified rate and saves them as PNG files in a 'frames' folder.

        :param video_path: Path to the video file.
        :return: List of file paths of the extracted frames.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

       # Create the frames directory based on the video name
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join("frames", video_name)
        
        
        os.makedirs(output_dir, exist_ok=True)

        # Open the video file
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate // self.frame_per_second)

        frame_count = 0
        saved_frames = []

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Save frames at the specified interval
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{frame_count}.png")
                cv2.imwrite(frame_filename, frame)
                saved_frames.append(frame_filename)

            frame_count += 1

        video_capture.release()
        return saved_frames

def main():
    """
    Test the FramesExtractor class here.
    """
    video_path = "videos/video1.mp4"
    frame_per_second = 2  # Example: extract 2 frames per second

    frames_extractor = FramesExtractor(frame_per_second)
    extracted_frames = frames_extractor(video_path)

    print(f"Extracted {len(extracted_frames)} frames. Frames saved in 'frames' folder.")

if __name__ == "__main__":
    main()
