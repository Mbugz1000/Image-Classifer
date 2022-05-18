import warnings
warnings.filterwarnings('ignore')

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import json
from PIL import Image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disabling the GPU

IMAGE_SIZE = 224


def load_model():
    dir_ = "data/for_modelling/tf2-preview_mobilenet_v2_feature_vector_4"
    feature_extractor = hub.KerasLayer(dir_, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    feature_extractor.trainable = False

    reloaded_model = tf.keras.models.load_model("data/for_modelling/final_model.h5",
                                                custom_objects={'KerasLayer': feature_extractor})

    return reloaded_model


def load_class_names():
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    return class_names


def process_image(image_numpy_array):
    image_tensor = tf.convert_to_tensor(image_numpy_array)
    image_tensor = tf.image.resize(image_tensor, size=(IMAGE_SIZE, IMAGE_SIZE))
    image_tensor /= 225

    return image_tensor


def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)

    output = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_k_idx = np.argsort(output[0])[-top_k:]

    return output[0][top_k_idx], np.arange(1, 103).astype(str)[top_k_idx]


def main(image_path, top_k):
    model = load_model()
    class_names = load_class_names()

    ps, classes = predict(image_path, model, top_k)

    print(f"The predicted image is a {class_names[classes[-1]]}. Predicted probability is {ps[-1]:.4f}\n")

    next_likely_flowers = list(map(class_names.get, classes[:-1]))
    print(f"The next {top_k-1} likely flowers according to the model are {', '.join(next_likely_flowers)}")


if __name__ == "__main__":
    print("*********************** Flower Predictor ***********************")
    image_path_ = input("Enter the image path: ")
    top_k_ = input("How many likely flowers do you want to see?: ")
    try:
        top_k_ = int(top_k_)
    except Exception as e:
        print("Kindly enter an integer")
        top_k_ = input("How many likely flowers do you want to see?: ")

    main(image_path_, top_k_)
