import cv2
import numpy as np
import skimage
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from src.logger import logger

# Define the image dimensions
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 128, 128, 3

def build_feature_extractor():
    input_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=input_layer, outputs=x)

def preprocess_images(imgs, image_size):
    images = []
    for img in imgs:
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, image_size)  # Add resizing here
        img = img / 255.0
        images.append(img)
    return np.array(images)

def build_model():
    # Build the Siamese network
    feature_extractor = build_feature_extractor()

    input_a = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    input_b = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    features_a = feature_extractor(input_a)
    features_b = feature_extractor(input_b)

    # Combine the features and classify
    merged = Concatenate()([features_a, features_b])
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[input_a, input_b], outputs=output)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

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
        self.shadow_tips = [4, 20]
        self.pad = 5

    def __find_lines_w(self, img):
        vals_left = img[:, 0:2 * self.pad]
        vals_left = vals_left < 150

        widths = [0]
        for i in range(img.shape[0]):
            cur_sum = sum(vals_left[i, :])
            if cur_sum < img.shape[1] / 7:
                widths.append(cur_sum)

        width_left = np.median(widths)

        vals_right = img[:, -2 * self.pad:]
        vals_right = vals_right < 150
        widths = [0]

        for i in range(img.shape[0]):
            cur_sum = sum(vals_right[i, :])
            if cur_sum < img.shape[1] / 7:
                widths.append(cur_sum)

        width_right = np.median(widths)
        return max(width_left, width_right)

    def create_train_data(self, white_keys_coords, black_keys_coords, frames, frame_without_hands):
        for i in range(len(self.coords_history)):
            for key in self.coords_history[i].keys():
                fingertip_idx = self.coords_history[i][key].idx
                ref_name = f"img/ref/{i}_{key}.png"
                cur_name = f"img/cur/{i}_{key}.png"

                # this is 1 or 5 finger we need to look at the shadow
                # in the edge of the key
                name_prob_pressed_key = self.coords_history[i][key].key_name
                if not "#" in name_prob_pressed_key:
                    y_ul, x_ul, y_dr, x_dr = white_keys_coords[name_prob_pressed_key].coords()

                else:
                    y_ul, x_ul, y_dr, x_dr = black_keys_coords[name_prob_pressed_key].coords()
                x_ul = x_ul - self.pad
                x_dr = x_dr + self.pad
                ref_key = cv2.cvtColor(frame_without_hands[y_ul:y_dr, x_ul:x_dr, ...], cv2.COLOR_RGB2GRAY)
                cur_img = cv2.cvtColor(frames[i][y_ul:y_dr, x_ul:x_dr, ...], cv2.COLOR_RGB2GRAY)

                skimage.io.imsave(ref_name, ref_key)
                skimage.io.imsave(cur_name, cur_img)


    def get_history(self, white_keys_coords, black_keys_coords, frames, frame_without_hands):
        for i in range(len(self.coords_history)):
            for key in self.coords_history[i].keys():
                fingertip_idx = self.coords_history[i][key].idx

                if not fingertip_idx in self.shadow_tips:
                    name = f"{self.coords_history[i][key].idx}_{self.coords_history[i][key].hand}"
                    max_dist = self.max_dists[name]
                    cur_dist = self.coords_history[i][key].dist
                    if cur_dist < max_dist * self.dist_ratio:
                        self.coords_history[i][key].set_press()
                else:
                    # this is 1 or 5 finger we need to look at the shadow
                    # in the edge of the key
                    name_prob_pressed_key = self.coords_history[i][key].key_name

                    if not "#" in name_prob_pressed_key:
                        y_ul, x_ul, y_dr, x_dr = white_keys_coords[name_prob_pressed_key].coords()

                    else:
                        y_ul, x_ul, y_dr, x_dr = black_keys_coords[name_prob_pressed_key].coords()
                    x_ul = x_ul - self.pad
                    x_dr = x_dr + self.pad
                    ref_key = cv2.cvtColor(frame_without_hands[y_ul:y_dr, x_ul:x_dr, ...], cv2.COLOR_RGB2GRAY)
                    cur_img = cv2.cvtColor(frames[i][y_ul:y_dr, x_ul:x_dr, ...], cv2.COLOR_RGB2GRAY)

                    max_w_ref = self.__find_lines_w(ref_key)
                    max_w_cur = self.__find_lines_w(cur_img)

                    if max_w_cur > max_w_ref:
                        # the button is pressed
                        self.coords_history[i][key].set_press()

                    # check if key has rectangular shadow
        return self.coords_history

    def __call__(self, keys_fingertips):
        # если отношение между расстоянием крайней и средней точкой
        # стало меньше чем максимальное расстояние
        # если это 1 или 5 палец то смотреть на тень

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

                            for i in range(len(self.coords_history) - 1, -1, -1):
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


class PressedKeyExtractorClassifyImg(PressedKeyExtractorBase):
    def __init__(self):
        super().__init__()
        self.coords_history = []
        self.model = self.__load_pretrained_model()

        self.pad = 5

    def __load_pretrained_model(self):
        pretrained_feature_extractor = build_feature_extractor()

        pretrained_input_a = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        pretrained_input_b = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

        pretrained_features_a = pretrained_feature_extractor(pretrained_input_a)
        pretrained_features_b = pretrained_feature_extractor(pretrained_input_b)

        pretrained_merged = Concatenate()([pretrained_features_a, pretrained_features_b])
        pretrained_output = Dense(1, activation='sigmoid')(pretrained_merged)

        pretrained_model = Model(inputs=[pretrained_input_a, pretrained_input_b], outputs=pretrained_output)
        pretrained_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        pretrained_model.load_weights('siamese_model.weights.h5')
        return pretrained_model

    def get_history(self, white_keys_coords, black_keys_coords, frames, frame_without_hands):
        ref_imgs = []
        cur_imgs = []
        idx = []

        for i in range(len(self.coords_history)):

            for key in self.coords_history[i].keys():

                name_prob_pressed_key = self.coords_history[i][key].key_name
                if not "#" in name_prob_pressed_key:
                    y_ul, x_ul, y_dr, x_dr = white_keys_coords[name_prob_pressed_key].coords()

                else:
                    y_ul, x_ul, y_dr, x_dr = black_keys_coords[name_prob_pressed_key].coords()
                x_ul = x_ul - self.pad
                x_dr = x_dr + self.pad

                ref_img = frame_without_hands[y_ul:y_dr, x_ul:x_dr, ...]
                cur_img = frames[i][y_ul:y_dr, x_ul:x_dr, ...]

                ref_imgs.append(ref_img)
                cur_imgs.append(cur_img)
                idx.append([i, key])

        X1 = preprocess_images(cur_imgs, (IMAGE_HEIGHT, IMAGE_WIDTH))
        X2 = preprocess_images(ref_imgs, (IMAGE_HEIGHT, IMAGE_WIDTH))


        pred = self.model.predict([X1, X2])
        pred = pred > 0.5

        for i in range(len(pred)):
            if pred[i]==1:
                self.coords_history[idx[i][0]][idx[i][1]].set_press()



        return self.coords_history

    def __call__(self, keys_fingertips):
        self.coords_history.append(keys_fingertips)

