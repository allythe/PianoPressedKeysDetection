import cv2
import numpy as np
import os

class FingerExtractor:
    def __init__(self):
        pass

    def extract_fingers(self, hand_mask):
        """
        Extract fingertip coordinates from the given hand mask using convex hull and local extrema methods.

        Args:
            hand_mask (numpy.ndarray): Binary mask of a hand.

        Returns:
            list: A list of fingertip coordinates for the hand mask.
        """
        # Preprocess mask to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.erode(processed_mask, kernel, iterations=1)

        fingertips = []
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)

            # Convexity Defects Method
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is not None and len(hull) > 3:
                try:
                    defects = cv2.convexityDefects(contour, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])
                            angle = self.calculate_angle(start, end, far)
                            if angle < np.pi / 2:  # Threshold to detect fingertip candidates
                                fingertips.append(start)
                                fingertips.append(end)
                except cv2.error as e:
                    print(f"Error in convexityDefects: {e}")

            # Local Extrema Method
            local_extrema = self.find_local_extrema(contour)
            fingertips.extend(local_extrema)

            # Filter unique fingertips by distance
            fingertips = self.filter_fingertips(fingertips)

        return fingertips

    def find_local_extrema(self, contour):
        """
        Find local extrema points on the contour.

        Args:
            contour (numpy.ndarray): Contour of the hand.

        Returns:
            list: List of local extrema points.
        """
        extrema = []
        for i in range(1, len(contour) - 1):
            prev = contour[i - 1][0]
            curr = contour[i][0]
            next = contour[i + 1][0]
            if curr[1] < prev[1] and curr[1] < next[1]:  # Detect local minima (potential fingertips)
                extrema.append(tuple(curr))
        return extrema

    def calculate_angle(self, pt1, pt2, pt3):
        """
        Calculate the angle formed by three points.

        Args:
            pt1, pt2, pt3 (tuple): Points in the format (x, y).

        Returns:
            float: Angle in radians.
        """
        a = np.linalg.norm(np.array(pt1) - np.array(pt3))
        b = np.linalg.norm(np.array(pt2) - np.array(pt3))
        c = np.linalg.norm(np.array(pt1) - np.array(pt2))
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
        return angle

    def filter_fingertips(self, points, min_distance=40):
        """
        Remove duplicate fingertip candidates based on a distance threshold.

        Args:
            points (list): List of fingertip points.
            min_distance (int): Minimum pixel distance to consider points unique.

        Returns:
            list: Filtered unique fingertip points.
        """
        filtered = []
        for pt in points:
            if all(np.linalg.norm(np.array(pt) - np.array(existing)) > min_distance for existing in filtered):
                filtered.append(pt)
        return filtered


def process_masks(video_name):
    """
    Process all hand masks for a video, extract fingertips, and visualize results.

    Args:
        video_name (str): Name of the video folder containing hand masks.

    Returns:
        None
    """
    masks_dir = f"masks/{video_name}"
    output_file = f"fingertips/{video_name}_fingertips.txt"
    visualization_dir = f"fingertips/visualizations/{video_name}"

    os.makedirs("fingertips", exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    if not os.path.exists(masks_dir):
        print("Error: Mask directory does not exist. Run the HandsExtractor first.")
        return

    finger_extractor = FingerExtractor()
    with open(output_file, "w") as f:
        for mask_name in sorted(os.listdir(masks_dir)):
            mask_path = os.path.join(masks_dir, mask_name)
            hand_mask = cv2.imread(mask_path, 0)
            if hand_mask is None:
                print(f"Warning: Could not load mask {mask_name}. Skipping.")
                continue

            fingertips = finger_extractor.extract_fingers(hand_mask)
            f.write(f"{mask_name}: {fingertips}\n")

            visualization = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
            for tip in fingertips:
                cv2.circle(visualization, tip, 5, (0, 0, 255), -1)

            visualization_path = os.path.join(visualization_dir, f"visual_{mask_name}")
            cv2.imwrite(visualization_path, visualization)

    print(f"All fingertip coordinates saved to {output_file}.")
    print(f"Visualizations saved to {visualization_dir}.")


if __name__ == "__main__":
    video_name = "video1"  # Replace with the name of your video folder
    process_masks(video_name)
