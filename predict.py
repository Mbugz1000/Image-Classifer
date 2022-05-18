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
    """
    This function loads the model as well as the pre-trained model for prediction
    :return: Keras model
    """
    dir_ = "data/for_modelling/tf2-preview_mobilenet_v2_feature_vector_4"
    feature_extractor = hub.KerasLayer(dir_, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    feature_extractor.trainable = False

    reloaded_model = tf.keras.models.load_model("data/for_modelling/final_model.h5",
                                                custom_objects={'KerasLayer': feature_extractor})

    return reloaded_model


def load_class_names():
    """

    :return: Dictionary that maps class labels to class names
    """
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    return class_names


def process_image(image_numpy_array):
    """
    This function resizes and normalizes the images

    :param image_numpy_array: Image in form of a numpy array
    :return: Transformed image tensor
    """
    image_tensor = tf.convert_to_tensor(image_numpy_array)
    image_tensor = tf.image.resize(image_tensor, size=(IMAGE_SIZE, IMAGE_SIZE))
    image_tensor /= 225

    return image_tensor


def predict(image_path, model, top_k):
    """
    This method predicts the flower that's in a given image.

    :param image_path: The path to the input image
    :param model: Trained model
    :param top_k: The top k most probable flowers in the list
    :return: Probabilities of the top k flowers and Labels of the top k flowers
    """
    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)

    output = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_k_idx = np.argsort(output[0])[-top_k:]

    return output[0][top_k_idx], np.arange(1, 103).astype(str)[top_k_idx]


def main(image_path, top_k):
    """
    The main method of this command line app. It performs the following tasks:
        1. Loads the model,
        2. Loads the class names
        3. Performs the prediction the returns the output to the user

    :param image_path: The path to the input image
    :param top_k: The top k most probable flowers in the list
    :return: None
    """
    model = load_model()
    class_names = load_class_names()

    ps, classes = predict(image_path, model, top_k)

    print(f"The predicted image is a {class_names[classes[-1]]}. Predicted probability is {ps[-1]:.4f}\n")

    next_likely_flowers = list(map(class_names.get, classes[:-1]))
    print(f"The next {top_k-1} likely flowers according to the model are:")
    for i, (flower, prob) in enumerate(zip(next_likely_flowers[::-1], ps[::-1][1:]), start=2):
        print(f"{i}. {flower}. Prob: {prob:.4f}")


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
