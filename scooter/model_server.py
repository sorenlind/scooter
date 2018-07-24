"""Scooter ML model server."""

import json
import time
import os

import numpy as np
import redis

# initialize constants used for server queuing
PREDICTION_QUEUE = "prediction:queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25

db = redis.StrictRedis(host=os.environ['SCOOTER_REDIS'], db=0)


def predictions_process(model, sample_decoder, prediction_decoder):
    """Continuously query queue for new prediction jobs and execute them."""

    # continually pool for new data to classify
    while True:
        batch_elements = db.lrange(PREDICTION_QUEUE, 0, BATCH_SIZE - 1)
        batch, x_ids = _build_batch(batch_elements, sample_decoder)

        if not x_ids:
            time.sleep(SERVER_SLEEP)
            continue

        # classify the batch
        print("Predicting on batch of size: %s" % (batch.shape,))
        preds = model.predict(batch)
        results = prediction_decoder(preds)

        # loop over the x IDs and their corresponding set of results from our model
        for (x_id, result_set) in zip(x_ids, results):
            # initialize the list of output predictions
            output = []

            # loop over the results and add them to the list of output predictions
            for (_, label, prob) in result_set:
                result = {"label": label, "probability": float(prob)}
                output.append(result)

            # store the predictions in the database, using the ID as the key so we can fetch the results
            db.set(x_id, json.dumps(output))

        # remove the set of images from our queue
        print("Removing %s item(s) from queue" % len(x_ids))
        db.ltrim(PREDICTION_QUEUE, len(x_ids), -1)

        time.sleep(SERVER_SLEEP)


def _build_batch(batch_elements, sample_decoder):
    # attempt to grab a batch of images from the database, then initialize the image IDs and batch of images themselves
    x_ids = []
    batch = None

    # loop over the queue
    for element in batch_elements:
        # deserialize the object and obtain the input image
        element = json.loads(element.decode("utf-8"))
        image = sample_decoder(element["x"])

        if batch is None:
            batch = image

        # otherwise, stack the data
        else:
            batch = np.vstack([batch, image])

        # update the list of image IDs
        x_ids.append(element["id"])
    return batch, x_ids


def start_model_server(model, decode_sample, decode_predictions):
    print("Starting prediction service")
    try:
        db.ping()
    except redis.ConnectionError:
        print("Cannot connect to redis. Aborting.")
        return
    predictions_process(model, decode_sample, decode_predictions)
