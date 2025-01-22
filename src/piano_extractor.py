import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class PianoExtractor:
    def __init__(self, manual=True, show_plots=False, mse_threshold=50):
        """
        Initializes the PianoExtractor.
        :param manual: If True, allows manual piano region selection.
        :param show_plots: If True, displays debugging plots.
        :param mse_threshold: Threshold for MSE to decide frame similarity.
        """
        self.manual = manual
        self.show_plots = show_plots
        self.mse_threshold = mse_threshold
        self.piano_region = None
        self.skin_cluster_counts = []

    def detect_piano_region_with_visualization(self, frame):
        """
        Automatically detects the piano region using preprocessing steps and visualizes each step.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame , (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 50, 150)

        # Apply dilation to enhance edges
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_rect = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Improved filtering criteria
            if 2 < aspect_ratio < 15 and w > 300 and h > 50 and y > 50:
                area = w * h
                if area > max_area and self.filter_piano_keys_by_color(frame, (x, y, w, h)):
                    max_area = area
                    best_rect = (x, y , w, h)

        if best_rect is None:
            raise ValueError("Piano region could not be detected automatically.")

        self.visualize_piano_detection_steps(frame, gray_frame, blurred_frame, dilated_edges, contours, best_rect)
        
        return best_rect,dilated_edges

    
    def filter_piano_keys_by_color(self, frame, region):
        """
        Filters detected region based on piano key colors.
        :param frame: Input video frame.
        :param region: Region of interest (x, y, w, h).
        :return: True if the region contains piano keys, else False.
        """
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)

        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_ratio = cv2.countNonZero(white_mask) / (w * h)

        # Check if the detected region has both black and white keys
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([180, 255, 50], dtype=np.uint8)
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        black_ratio = cv2.countNonZero(black_mask) / (w * h)

        total_ratio = white_ratio + black_ratio

        # Visualize the detected region and mask
        if self.show_plots:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            plt.title("Detected Region (ROI)")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(white_mask + black_mask, cmap='gray')
            plt.title(f"White/Black Key Mask (Coverage: {total_ratio:.2f})")
            plt.axis("off")
            plt.suptitle("Piano Key Color Detection", fontsize=16)
            plt.show()

        return white_ratio > 0.15 and black_ratio > 0.05  # Ensure detection includes both black and white keys

    def refine_piano_region_with_pattern(self, frame, dilated_edges, best_rect):
        """
        Refines the detected piano region by identifying key patterns in the dilated edge image.
        :param frame: Input video frame.
        :param dilated_edges: Edge-detected and dilated image.
        :param best_rect: Initial detected piano bounding box (x, y, w, h).
        :return: Refined bounding box (x, y, w, h).
        """
        x, y, w, h = best_rect
        roi = dilated_edges[y:y+h, x:x+w]

        # Step 1: Perform vertical projection
        vertical_projection = np.sum(roi, axis=0)

        # Step 2: Detect peaks (white key boundaries)
        threshold = np.max(vertical_projection) * 0.5
        key_positions = np.where(vertical_projection > threshold)[0]

        if len(key_positions) > 0:
            new_x = key_positions[0] + x
            new_w = key_positions[-1] - key_positions[0]
        else:
            new_x, new_w = x, w

        # Step 3: Apply a horizontal projection to further refine the region
        horizontal_projection = np.sum(roi, axis=1)
        key_rows = np.where(horizontal_projection > np.max(horizontal_projection) * 0.5)[0]

        if len(key_rows) > 0:
            new_y = key_rows[0] + y
            new_h = key_rows[-1] - key_rows[0]
        else:
            new_y, new_h = y, h

        refined_rect = (new_x, new_y, new_w, new_h)

        # Visualization of the refined region
        cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 255), 2)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
        plt.title("Original Detected Region")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Refined Piano Region")
        plt.axis("off")
        plt.show()

        return refined_rect




    def visualize_piano_detection_steps(self, frame, gray_frame, blurred_frame, edges, contours, piano_region):
        """
        Visualizes different stages of the piano region detection process.
        """
        x, y, w, h = piano_region
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.figure(figsize=(15, 8))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Original Frame")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(gray_frame, cmap='gray')
        plt.title("Grayscale Conversion")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(blurred_frame, cmap='gray')
        plt.title("Gaussian Blur")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(edges, cmap='gray')
        plt.title("Dilated Edge Detection")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        contours_frame = np.zeros_like(gray_frame)
        cv2.drawContours(contours_frame, contours, -1, (255, 255, 255), 1)
        plt.imshow(contours_frame, cmap='gray')
        plt.title("Contour Detection")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        plt.title("Final Detected Region")
        plt.axis("off")

        plt.suptitle("Piano Detection Process", fontsize=16)
        plt.show()




    def get_piano_region(self, frame):
        """
        Determines the piano region either manually or automatically.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        if self.manual:
            print("Selecting piano region manually...")
            return self.select_piano_region(frame)
        else:
            best_rect,dialeted_edges = self.detect_piano_region_with_visualization(frame)
            refined_rect = self.refine_piano_region_with_pattern(frame, dialeted_edges, best_rect)
            return refined_rect

    def select_piano_region(self, frame):
        """
        Allows the user to manually select the piano region in a frame.
        :param frame: Input video frame.
        :return: Coordinates of the piano region (x, y, w, h).
        """
        roi = cv2.selectROI("Select Piano Region", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Piano Region")
        return roi

    def calculate_mse(self, frame1, frame2):
        """
        Calculate Mean Squared Error (MSE) between two frames.
        :param frame1: First frame (grayscale).
        :param frame2: Second frame (grayscale).
        :return: Mean Squared Error value.
        """
        return np.mean((frame1.astype("float") - frame2.astype("float")) ** 2)

    def detect_skin_clusters(self, frame):
        """
        Detects skin pixel clusters using contour analysis within the selected piano region.
        :param frame: Input video frame.
        :return: Number of detected skin clusters in the piano region.
        """
        x, y, w, h = self.piano_region
        roi_frame = frame[y:y+h, x:x+w]
        ycrcb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)

        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        skin_mask = cv2.inRange(ycrcb_frame, lower_skin, upper_skin)

        # Find contours of skin regions
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cluster_count = len([c for c in contours if cv2.contourArea(c) > 500])  # Count clusters with significant size

        self.skin_cluster_counts.append(cluster_count)

        """if self.show_plots:
            plt.figure(figsize=(6, 6))
            plt.imshow(skin_mask, cmap='gray')
            plt.title(f"Detected Skin Clusters: {cluster_count}")
            plt.axis("off")
            plt.show()"""

        return cluster_count

    def extract_clear_frame(self, frames):
        """
        Extract the first frame where no hands are present on the piano.
        :param frames: List of frames (numpy arrays).
        :return: Index of the first clear frame.
        """
        if self.piano_region is None:
            self.piano_region = self.get_piano_region(frames[0])
            print(f"Selected piano region: {self.piano_region}")

        previous_frame = None
        mse_values = []
        clear_frame_idx = None

        for idx, frame in enumerate(frames):
            x, y, w, h = self.piano_region
            cropped_gray_frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

            # Calculate skin clusters in the selected region
            skin_clusters = self.detect_skin_clusters(frame)

            if previous_frame is not None:
                mse = self.calculate_mse(previous_frame, cropped_gray_frame)
                mse_values.append(mse)

                # Condition to identify a clear frame (low MSE + minimal skin clusters)
                if mse < self.mse_threshold and skin_clusters == 0:
                    clear_frame_idx = idx
                    break

            previous_frame = cropped_gray_frame

        if self.show_plots:
            self.plot_metrics(mse_values)

        if clear_frame_idx is None:
            raise ValueError("No clear piano frame found.")
        
        return clear_frame_idx

    def plot_metrics(self, mse_values):
        """
        Plots the MSE values and skin cluster counts across all frames.
        :param mse_values: List of computed MSE values.
        """
        plt.figure(figsize=(12, 6))

        # Plot MSE values
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o', label='MSE Values')
        plt.axhline(y=self.mse_threshold, color='r', linestyle='--', label=f'Threshold ({self.mse_threshold})')
        plt.xlabel('Frame Index')
        plt.ylabel('MSE Value')
        plt.title('MSE of Cropped Regions')
        plt.legend()
        plt.grid(True)

        # Plot skin cluster counts
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.skin_cluster_counts)), self.skin_cluster_counts, marker='o', color='orange', label='Skin Clusters')
        plt.axhline(y=0, color='r', linestyle='--', label='Expected Clear Frame')
        plt.xlabel('Frame Index')
        plt.ylabel('Skin Cluster Count')
        plt.title('Skin Clusters in Selected Region')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    # Load frames from a directory into a list
    frames_dir = "C:\\Users\\Friday\\Desktop\\IACV PROJECT\\PianoPressedKeysDetection\\frames2\\video1\\cleaned"
    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    frames = [cv2.imread(frame_path) for frame_path in frame_paths]

    # Initialize the PianoExtractor
    extractor = PianoExtractor(manual=False, show_plots=True, mse_threshold=50)

    # Extract a clear piano frame from the frames list
    try:
        clear_frame_idx = extractor.extract_clear_frame(frames)
        print(f"Clear piano frame found at index: {clear_frame_idx}")

        # Save the clear frame for verification
        cv2.imwrite(f"clear_piano_frame_{clear_frame_idx}.png", frames[clear_frame_idx])
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
