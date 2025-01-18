import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import xml.etree.ElementTree as ET

# Define the image dimensions
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 128, 128, 3


# Function to build the feature extraction model
def build_feature_extractor():
    input_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return Model(inputs=input_layer, outputs=x)


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


# Prepare the dataset
# Assume X1, X2 are the two sets of images and Y is the label (1 for pressed, 0 for not pressed)
def preprocess_images(image_paths, image_size):
    images = []
    for path in image_paths:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, image_size)  # Add resizing here
        img = img / 255.0
        images.append(img)
    return np.array(images)


def load_data():
    # Read XML content from a file
    file_path = 'annotations.xml'  # Replace with the path to your XML file
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Parse the XML content
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    # Extract image names and 'Pressed' states
    results = []
    for image in root.findall(".//image"):
        name = image.get("name")
        pressed_state = image.find(".//attribute[@name='Pressed']").text
        results.append((name, pressed_state))

    X1_paths = []
    X2_paths = []
    Y = []

    # Print the extracted results
    for name, pressed in results:
        # print(f"Image: {name}, Pressed: {pressed}")
        X1_paths.append(f"img/cur/{name}")
        X2_paths.append(f"img/ref/{name}")
        if pressed == "false":
            Y.append(0)
        else:
            Y.append(1)

    X1 = preprocess_images(X1_paths, (IMAGE_HEIGHT, IMAGE_WIDTH))
    X2 = preprocess_images(X2_paths, (IMAGE_HEIGHT, IMAGE_WIDTH))
    Y = np.array(Y)

    print(f"Number of X1 {X1.shape}")
    print(f"Number of X2 {X2.shape}")
    print(f"Number of Y {len(Y)}")

    return X1, X2, Y


def train_model():
    model = build_model()
    print(model.summary())
    X1, X2, Y = load_data()

    # Train the model
    model.fit([X1, X2], Y, batch_size=16, epochs=10, validation_split=0.2)

    # Save the model weights
    model.save_weights('siamese_model.weights.h5')


# Load the model weights
def load_pretrained_model():
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


# Function for inference
def make_inference(X1, X2):
    pretrained_model = load_pretrained_model()

    prediction = pretrained_model.predict([X1, X2])
    return prediction  # Return the predicted probability


def main():
    train_model()
    X1, X2, Y = load_data()
    pred = make_inference(X1, X2)
    pred = pred > 0.5
    acc = []
    for i in range(len(pred)):
        if pred[i] == Y[i]:
            acc.append(1)
        else:
            acc.append(0)

    print(np.mean(acc))


if __name__ == "__main__":
    main()
