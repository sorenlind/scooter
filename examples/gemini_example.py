"""Example showing how to build an API on top of Gemini using Scooter."""
import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from gemini.cards import read_card_list
from gemini.image import flatten
from scooter.model_server import start_model_server
from scooter.web_server import start_web_server

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 448
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"


class Classifier(object):
    """Class for classifying Magic cards from photos."""

    def __init__(self, cards, knn, encoder, graph):
        """Initialize a classifier."""
        self._cards = cards
        self._knn = knn
        self._encoder = encoder
        self._graph = graph
        self._encoded_dim = np.prod(self._encoder.output.shape[1:]).value

    def predict(self, batch):
        # TODO: Clean this up
        print("encoding user image")
        encoded = np.zeros((1, self._encoded_dim))
        with self._graph.as_default():
            batch_encoded = self._encoder.predict(batch)
        batch_encoded_flat = flatten(batch_encoded)
        encoded[0, :] = batch_encoded_flat
        print(encoded.shape)

        print("finding most similar")
        most_similar = self._knn.kneighbors(encoded, return_distance=True)
        print("found most similar")
        print("batch shape:", batch.shape)
        print("most_similar length:", len(most_similar))

        results = []
        for i in range(batch.shape[0]):
            scores = most_similar[0][i]
            indices = most_similar[1][i]
            k_val = scores.shape[0]
            temp_results = []
            for top_pos in range(k_val):
                card_index = indices[top_pos]
                temp = (top_pos + 1, self._cards[card_index].name + " / " + self._cards[card_index].edition,
                        scores[top_pos])
                temp_results.append(temp)
            results.append(temp_results)
        print("results length", len(results))
        print("returning results")
        return results


def load_classifier(card_list_file, encodings_file, encoder_file):
    print("loading cards")
    cards = read_card_list(card_list_file)

    print("loading encodings")
    encoded_img = np.load(encodings_file)
    print("encodings shape: %s", encoded_img.shape)

    print("fitting knn")
    knn = NearestNeighbors(n_neighbors=3, algorithm="brute", metric="cosine")
    knn.fit(encoded_img)

    print("loading encoder model")
    model = load_model(encoder_file)
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoder_output').output)
    graph = tf.get_default_graph()

    print("classifier loaded")

    return Classifier(cards, knn, encoder, graph)


def load_gemini_model():
    print("Loading Gemini model...")
    card_list_file = "/users/sorenlind/Data/mtg/cards_en.csv"
    encodings_file = "/users/sorenlind/Data/mtg/encodings/encoded_master_v03_no_aug.npy"
    encoder_file = "/users/sorenlind/Data/mtg/models/gcp/autoencoder_master_v03_no_aug/autoencoder_master_v03_no_aug"
    model = load_classifier(card_list_file, encodings_file, encoder_file)
    print("Gemini model loaded")

    return model


def decode_sample(base64_image):
    loaded_image_data = Image.open(BytesIO(base64.b64decode(base64_image)))
    assert loaded_image_data.size == (320, 448)
    image = img_to_array(loaded_image_data)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.
    return image


def decode_predictions(predictions):
    #return imagenet_utils.decode_predictions(predictions)
    return predictions


if __name__ == "__main__":
    start_model_server(load_gemini_model, decode_sample, decode_predictions)
    start_web_server(debug=None)
