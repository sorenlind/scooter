"""Scooter ML model server."""

import json
import logging
import time

import numpy as np
import redis

from scooter.settings import REDIS_HOST, REDIS_QUEUE, WORKER_SLEEP, BATCH_SIZE

logger = logging.getLogger(__name__)

db = redis.StrictRedis(host=REDIS_HOST, db=0)


def predictions_process(model, sample_decoder, prediction_decoder):
    """Continuously query queue for new prediction jobs and execute them."""
    while True:
        pipe = db.pipeline()
        pipe.lrange(REDIS_QUEUE, 0, BATCH_SIZE - 1)
        pipe.ltrim(REDIS_QUEUE, BATCH_SIZE, -1)
        batch_elements, _ = pipe.execute()

        batch, x_ids, bad_ids = _build_batch(batch_elements, sample_decoder)

        for bad_id in bad_ids:
            db.set(bad_id, json.dumps({"error": "Unknown"}))

        if not x_ids:
            time.sleep(WORKER_SLEEP)
            continue

        logger.info("Predicting on batch of size: %s", (batch.shape, ))
        preds = model.predict(batch)
        results = prediction_decoder(preds)

        for (x_id, result_set) in zip(x_ids, results):
            output = []
            for (_, label, prob) in result_set:
                result = {"label": label, "probability": float(prob)}
                output.append(result)

            db.set(x_id, json.dumps(output))


def _build_batch(batch_elements, sample_decoder):
    x_ids = []
    batch = None
    bad_ids = []

    for element in batch_elements:
        element = json.loads(element.decode("utf-8"))
        try:
            image = sample_decoder(element["x"])
        except Exception:
            logger.info("Received bad data. Cannot put on queue.")
            bad_ids.append(element["id"])
            continue

        if batch is None:
            batch = image

        else:
            batch = np.vstack([batch, image])

        x_ids.append(element["id"])
    return batch, x_ids, bad_ids


def start_model_server(model, decode_sample, decode_predictions):
    logger.info("Starting prediction service")
    try:
        db.ping()
    except redis.ConnectionError:
        logger.error("Cannot connect to redis. Aborting.")
        return
    logger.info("Ready for prediction jobs")
    predictions_process(model, decode_sample, decode_predictions)
