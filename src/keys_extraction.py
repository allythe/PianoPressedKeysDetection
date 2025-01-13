from copy import deepcopy

import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.io
from skimage.feature import match_template
from sklearn.cluster import DBSCAN

from src.logger import logger


class Key:
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        self.y_ul = y_ul
        self.x_ul = x_ul
        self.y_dr = y_dr
        self.x_dr = x_dr

        self.key_img = key_img
        self.name = name

    def coords(self):
        """

        :return: y_ul, x_ul, y_dr, x_dr
        """
        return self.y_ul, self.x_ul, self.y_dr, self.x_dr

    def image(self):
        return self.key_img

    def __str__(self):
        return self.name


class WhiteKey(Key):
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        super().__init__(key_img, y_ul, x_ul, y_dr, x_dr, name)
        if self.name[0] in "CDFGA":
            self.black_is_next = True
        else:
            self.black_is_next = False


class BlackKey(Key):
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        super().__init__(key_img, y_ul, x_ul, y_dr, x_dr, name)


key_names = {
    0: "C",
    1: "D",
    2: "E",
    3: "F",
    4: "G",
    5: "A",
    6: "B",
}


class KeysExtractorThroughLines:
    def __init__(self):
        self.logger = logger

        ref_piano_path = "src/octava.png"
        ref_piano = skimage.io.imread(ref_piano_path)
        self.ref_piano = cv2.cvtColor(ref_piano, cv2.COLOR_RGB2GRAY)
        self.logger.info("Keys Extractor created")

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _find_piano_contour(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = cv2.adaptiveThreshold(image_gray, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)

        return image_gray, c

    def _draw_contour(self, image, c):
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return x, y, w, h

    def _create_piano_masked(self, image, c):
        mask = np.zeros_like(image)
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        image[mask == 0] = 0

        return image

    def _find_contour_angle(self, image, c):
        center, hw, angle = cv2.minAreaRect(c)
        h, w = hw

        if w > h:
            angle = 90 + angle

        self.logger.debug(f"Found h of contour is {np.round(h, 2)} px, width is {np.round(w, 2)} px")
        self.logger.debug(f"Found angle of contour is {np.round(angle, 2)} degrees")

        return angle
        # box = cv2.boxPoints(rect)

    def _find_kp(self, image):
        sift = cv2.SIFT.create()
        kp, des = sift.detectAndCompute(image, None)
        return kp, des

    def build_homography(self, img1, kp1, des1, img2, kp2, des2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, 255, 3, cv2.LINE_AA)

        else:
            self.logger.debug("Not enough matches are found - {}/{}".format(len(good), 10))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        plt.imshow(img3, 'gray'), plt.show()

    def _calculate_similarity(self, frame1, frame2):
        h1, w1 = frame1.shape
        h2, w2 = frame2.shape

        h_min = min(h1, h2)
        w_min = min(w1, w2)

        frame1 = deepcopy(frame1[:h_min, :w_min, ...])
        frame2 = deepcopy(frame2[:h_min, :w_min, ...])

        frame1[frame1 > 127] = 255
        frame1[frame1 <= 127] = 0

        frame2[frame2 > 127] = 255
        frame2[frame2 <= 127] = 0

        black1 = np.sum(frame1 == 0)
        black2 = np.sum(frame2 == 0)

        white1 = np.sum(frame1 == 255)
        white2 = np.sum(frame2 == 255)

        eps = 1e-9
        score = (black1 / (white1 + eps)) / (black2 / (white2 + eps) + eps)

        if score > 1:
            score = 1 / score

        return score

    def _find_white_keys_coords(self, masked_piano, x, y, w, h, white_key_w):
        keys_dict = {}
        patch = masked_piano[y:y + h, x:x + w, ...]
        patch = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 5)

        ref_piano_h, ref_piano_w = self.ref_piano.shape[:2]
        ref_piano_new_w = int(white_key_w * 7)
        ref_piano_new_h = int(ref_piano_h * (ref_piano_new_w / ref_piano_w))
        ref_piano_resh = cv2.resize(self.ref_piano, (ref_piano_new_w, ref_piano_new_h))

        result = match_template(patch, ref_piano_resh)
        res_max = np.max(result)

        mask = np.zeros_like(result)
        mask[result > res_max * 0.85] = 1

        peaks_coords_y, peaks_coords_x = np.where(mask == 1)

        peaks_coords = np.array(list(zip(peaks_coords_y, peaks_coords_x)))
        clustering_octave = DBSCAN(eps=3, min_samples=2).fit(peaks_coords)
        labels_octave = clustering_octave.labels_

        # left upper coordinates of the "do(C)" key in each found octave
        octave_coords_lu = []
        for label in np.unique(labels_octave):
            coord_y = int(np.mean(peaks_coords_y[labels_octave == label])) + y
            coord_x = int(np.mean(peaks_coords_x[labels_octave == label])) + x

            for j in range(20):
                if np.mean(masked_piano[coord_y, coord_x - 5:coord_x + 5]) - np.mean(
                        masked_piano[coord_y - j, coord_x - 5:coord_x + 5]) > 100:
                    coord_y = coord_y - j
                    break
            octave_coords_lu.append([coord_y, coord_x])

        octave_coords_lu.sort(key=lambda x: x[1])
        diff = [octave_coords_lu[i + 1][1] - octave_coords_lu[i][1] for i in range(len(octave_coords_lu) - 1)]
        diff.append(diff[-1])

        octave_coords_lu = np.array(octave_coords_lu)
        img_h, img_w = masked_piano.shape[:2]

        y_coord_bottom_keys = []
        for coord in octave_coords_lu:
            y, x = coord
            for j in range(y, img_h):
                if np.mean(masked_piano[j:j + 3, x - 5:x + 5]) / np.mean(masked_piano[j + 3:j + 6, x - 5:x + 5]) > 2:
                    y_coord_bottom_keys.append(j + 3)
                    break
        h_keys = int(np.median(y_coord_bottom_keys) - np.median(octave_coords_lu[:, 0]))

        self.logger.info(f"Found height of white keys is {h_keys}")

        all_white_keys_coords = []
        num_octave = 1
        # add coordinates of keys in found octaves
        for idx, coord in enumerate(octave_coords_lu):
            y, x = coord
            w_keys = diff[idx] / 7

            for i in range(7):
                name_cur = f"{key_names[i]}_{num_octave}"
                key_cur = WhiteKey(masked_piano[y: y + h_keys, int(x + i * w_keys):int(x + (i + 1) * w_keys)],
                                   y, int(x + i * w_keys), y + h_keys, int(x + (i + 1) * w_keys), name_cur)
                keys_dict[name_cur] = key_cur
                all_white_keys_coords.append([y, int(x + i * w_keys), y + h_keys, int(x + (i + 1) * w_keys)])

            num_octave += 1

        # add coordinates of keys not in full octave,
        # going before the first octave
        start_y, start_x, _, _ = keys_dict["C_1"].coords()
        for j in range(14):
            if int(start_x - white_key_w * (j % 7 + 1)) + 5 < 0:
                break

            patch_key = masked_piano[start_y: start_y + h_keys - 1,
                        int(start_x - white_key_w * (j % 7 + 1)): int(start_x - white_key_w * (j % 7))]
            name_key = key_names[6 - j % 7]
            true_key = keys_dict[f"{name_key}_2"].image()

            score = self._calculate_similarity(patch_key, true_key)

            if score < 0.5:
                break

            name_cur = f"{name_key}_0"
            key_cur = WhiteKey(patch_key, start_y, int(start_x - white_key_w * (j % 7 + 1)),
                               start_y + h_keys - 1,
                               int(start_x - white_key_w * (j % 7)),
                               name_cur)
            keys_dict[name_cur] = key_cur

        # add coordinates of keys not in full octave,
        # going after the last octave
        start_y, _, _, start_x = keys_dict[f"B_{num_octave - 1}"].coords()
        for j in range(14):
            if int(start_x + white_key_w * (j % 7 + 1)) - 5 > img_w:
                break

            patch_key = masked_piano[start_y: start_y + h_keys - 1,
                        int(start_x + white_key_w * (j % 7)): int(start_x + white_key_w * ((j % 7) + 1))]

            name_key = key_names[j]
            true_key = keys_dict[f"{name_key}_2"].image()

            score = self._calculate_similarity(patch_key, true_key)
            if score < 0.5:
                # it can be the last "C" which looks different from usual "C"
                if np.sum(patch_key > 200) / (patch_key.shape[0] * patch_key.shape[1]) < 0.8:
                    break

            name_cur = f"{name_key}_{num_octave}"
            key_cur = WhiteKey(patch_key, start_y, int(start_x + white_key_w * (j % 7)),
                               start_y + h_keys - 1,
                               int(start_x + white_key_w * (j % 7 + 1)),
                               name_cur)
            keys_dict[name_cur] = key_cur

        return keys_dict, h_keys

    def _find_black_keys_w(self, masked_piano, x, y, w, h, white_key_h, white_keys_coords):
        patch = masked_piano[y:y + h, x:x + w, ...]
        patch = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 5)

        black_key_y = int(white_keys_coords["C_1"].coords()[0] - y + white_key_h / 3)
        black_w_list = []

        cur_len = 0
        for j in range(w):
            if patch[black_key_y, j] == 0:
                cur_len += 1
            else:
                if cur_len > 0:
                    black_w_list.append(cur_len)
                cur_len = 0

        black_w = np.median(black_w_list)

        return black_w

    def _find_black_keys_coords(self, masked_piano, black_w, white_keys_coords):

        # using the info about the location of black keys (between white),
        # derive their coordinates

        white_keys = list(white_keys_coords.keys())
        black_keys = {}
        white_keys.sort(key=lambda x: 100 * int(x[-1]) + "CDEFGAB".index(x[0]))

        masked_piano[masked_piano > 150] = 255
        masked_piano[masked_piano < 150] = 0

        for key_name in white_keys:
            white_key = white_keys_coords[key_name]

            if white_key.black_is_next:
                name_cur = white_key.name
                name_cur = name_cur[0] + "#" + name_cur[1:]

                y_ul, _, y_dr, x_dr = white_key.coords()
                y_ul_b = y_ul
                x_ul_b = int(x_dr - black_w // 2)

                x_dr_b = int(x_ul_b + black_w)
                x_mean_b = int((x_dr_b + x_ul_b) / 2)

                y_dr_b = 0
                for i in range(int((y_dr - y_ul) / 3), y_dr - y_ul):

                    if abs(np.mean(masked_piano[y_ul + i, x_mean_b - 2:x_mean_b + 2]) - np.mean(
                            masked_piano[y_ul + i + 1, x_mean_b - 2:x_mean_b + 2])) > 20:
                        y_dr_b = y_ul + i
                        break

                max_dif = 15

                for i in range(-int(black_w / 4), int(black_w / 4)):

                    cur_dif = abs(np.mean(masked_piano[y_ul + 5:y_ul + 10, x_ul_b + i - 1: x_ul_b + i]) - np.mean(
                        masked_piano[y_ul + 5:y_ul + 10, x_ul_b + i: x_ul_b + i + 1]))

                    if cur_dif > max_dif:
                        max_dif = cur_dif
                        x_ul_b = x_ul_b + i

                max_dif = 15
                for i in range(-int(black_w / 4), int(black_w / 4)):

                    cur_dif = abs(np.mean(masked_piano[y_ul + 5:y_ul + 10, x_dr_b + i - 1: x_dr_b + i]) - np.mean(
                        masked_piano[y_ul + 5:y_ul + 10, x_dr_b + i: x_dr_b + i + 1]))

                    if cur_dif > max_dif:
                        max_dif = cur_dif
                        x_dr_b = x_dr_b + i
                if y_dr_b>0:
                   black_key = BlackKey(masked_piano[y_ul_b: y_dr_b, x_ul_b:x_dr_b],
                                         y_ul_b, x_ul_b, y_dr_b, x_dr_b, name_cur)
                   black_keys[name_cur] = black_key

        for key in black_keys.keys():
            print(black_keys[key].coords())
        return black_keys

    def _find_white_keys_w(self, image):
        mask = cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        lines = cv2.HoughLines(mask, 1, np.pi / 180, 150, min_theta=0, max_theta=0.01)
        cnt = 0
        x_coords = []
        for r_theta in lines:
            cnt += 1
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            x_coords.append(x1)
            cv2.line(mask, (x1, y1), (x2, y2), (255), 2)

        x_coords = np.sort(x_coords)
        x_coords_dif = [x_coords[i + 1] - x_coords[i] for i in range(len(x_coords) - 1)]
        white_key_w = np.median(x_coords_dif)

        self.logger.info(f"The width of white key is {white_key_w}")
        return image, white_key_w


    def _find_orientation(self, image, y, h):
        mask = cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        rows_max = np.mean(mask, axis=1)
        rows_piano = rows_max[y:y + h]
        first_part_mean = np.mean(rows_piano[:len(rows_piano) // 2])
        second_part_mean = np.mean(rows_piano[len(rows_piano) // 2:])

        if second_part_mean < first_part_mean:
            return False

        return True

    def __call__(self, image: np.ndarray):
        """
        this method will return coordinates of white
        and black keys in the current image
        (the image should be without hands!) """

        # simulate rotated image
        # image = self.rotate_image(image, 25)

        to_draw_img = deepcopy(image)

        # find the initial piano contour
        image_gray, c = self._find_piano_contour(image)

        # define the rotation angle of the contour
        angle = self._find_contour_angle(image_gray, c)

        # rotate image on this angle
        image = self.rotate_image(image, angle)
        to_draw_img = self.rotate_image(to_draw_img, angle)

        # find the new (angle is 0) piano contour
        image_gray, c = self._find_piano_contour(image)

        # draw piano contour on image
        x, y, w, h = self._draw_contour(image, c)

        # delete irrelevant info from image
        masked_piano = self._create_piano_masked(image_gray, c)

        # if the piano is upside-down, it should be rotated by 180 degrees
        is_correct_orientation = self._find_orientation(masked_piano, y, h)

        if not is_correct_orientation:
            masked_piano = self.rotate_image(masked_piano, 180)
            to_draw_img = self.rotate_image(to_draw_img, 180)
            angle += 180

        # find width of the white key
        masked_piano, white_key_w = self._find_white_keys_w(masked_piano)

        # find coordinates of white keys
        white_keys_coords, white_key_h = self._find_white_keys_coords(masked_piano, x, y, w, h, white_key_w)
        # self._draw_keys_coords(white_keys_coords, to_draw_img)

        # find h and w of black keys
        black_key_w = self._find_black_keys_w(masked_piano, x, y, w, h, white_key_h, white_keys_coords)
        black_keys_coords = self._find_black_keys_coords(masked_piano, black_key_w, white_keys_coords)

        return white_keys_coords, black_keys_coords, angle


def main():
    """
    test KeysExtractorThroughLines and other extractors here
    :return:
    """
    keys_extractor = KeysExtractorThroughLines()
    img_path = "../src/frames/video1/frame_7.png"
    image = skimage.io.imread(img_path)

    keys_extractor(image)


if __name__ == "__main__":
    main()
