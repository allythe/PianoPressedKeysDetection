import cv2
import numpy as np
import os

class FingerExtractor:
    def __init__(self):
        pass

    def extract_fingers(self, hand_mask):
        """
        Extract fingertip coordinates from the given hand mask with noise reduction.

        Args:
            hand_mask (numpy.ndarray): Binary mask of a hand.

        Returns:
            list: A list of fingertip coordinates for the hand mask.
        """
        # Preprocess mask to reduce noise
        kernel = np.ones((5, 5), np.uint8)  # Adjust size as needed
        processed_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        processed_mask = cv2.erode(processed_mask, kernel, iterations=1)  # Remove small noise

        fingertips = []
        # Find contours of the processed mask
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Get the largest contour (assumed to be the hand)
            contour = max(contours, key=cv2.contourArea)

            # Convexity Defects Method
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is not None and len(hull) > 3:
                # Check if the contour is valid and convex
                try:
                    defects = cv2.convexityDefects(contour, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])
                            angle = self.calculate_angle(start, end, far)
                            if angle < np.pi / 2:  # Example threshold for angle
                                fingertips.append(start)
                except cv2.error as e:
                    print(f"Error in convexityDefects: {e}")
                    # Skip this contour if it's invalid
                    pass

            # Local Extrema Method
            local_extrema = self.find_local_extrema(contour)
            for point in local_extrema:
                fingertips.append(point)

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
            if curr[1] < prev[1] and curr[1] < next[1]:  # Local minima (adjust as needed)
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


def process_masks(video_name):
    """
    Process all hand masks for a video, extract fingertips, and visualize results.

    Args:
        video_name (str): Name of the video folder containing hand masks.

    Returns:
        None
    """
    # Paths
    masks_dir = f"masks/{video_name}"
    output_file = f"fingertips/{video_name}_fingertips.txt"
    visualization_dir = f"fingertips/visualizations/{video_name}"

    if not os.path.exists(masks_dir):
        print("Error: Mask directory does not exist. Run the HandsExtractor first.")
        return

    if not os.path.exists("fingertips"):
        os.makedirs("fingertips")

    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # Initialize the FingerExtractor
    finger_extractor = FingerExtractor()

    # Process each mask in the masks directory
    with open(output_file, "w") as f:
        for mask_name in sorted(os.listdir(masks_dir)):
            mask_path = os.path.join(masks_dir, mask_name)
            hand_mask = cv2.imread(mask_path, 0)
            if hand_mask is None:
                print(f"Warning: Could not load mask {mask_name}. Skipping.")
                continue

            # Extract the fingertips
            fingertips = finger_extractor.extract_fingers(hand_mask)

            # Write fingertips to the file
            f.write(f"{mask_name}: {fingertips}\n")

            # Visualize the fingertips on the mask
            visualization = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
            for tip in fingertips:
                cv2.circle(visualization, tip, 5, (0, 0, 255), -1)  # Draw red circles for fingertips

            # Save the visualization
            visualization_path = os.path.join(visualization_dir, f"visual_{mask_name}")
            cv2.imwrite(visualization_path, visualization)

    print(f"All fingertip coordinates saved to {output_file}.")
    print(f"Visualizations saved to {visualization_dir}.")


if __name__ == "__main__":
    video_name = "video1"  # Replace with the name of your video folder
    process_masks(video_name)
