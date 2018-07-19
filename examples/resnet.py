"""Example showing how to build an API on top of ResNet50 using Scooter."""
import base64
from io import BytesIO

import numpy as np
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array
from PIL import Image

from scooter.web_server import start_web_server
from scooter.model_server import start_model_server

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"


def load_model():
    print("Loading ResNet50 model...")
    model = ResNet50(weights="imagenet")
    print("ResNet50 model loaded")

    return model


def decode_sample(base64_image):
    loaded_image_data = Image.open(BytesIO(base64.b64decode(base64_image)))

    if loaded_image_data.mode != "RGB":
        loaded_image_data = loaded_image_data.convert("RGB")

    loaded_image_data = loaded_image_data.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = img_to_array(loaded_image_data)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


def decode_predictions(predictions):
    return imagenet_utils.decode_predictions(predictions)


if __name__ == "__main__":
    start_model_server(load_model, decode_sample, decode_predictions)
    start_web_server(debug=None)
