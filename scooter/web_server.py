"""Scooter RESTful API."""

import logging
import json
import os
import socket
import time
import uuid

import flask
from redis import RedisError, StrictRedis

from scooter.settings import REDIS_HOST, REDIS_QUEUE, WEB_SLEEP


app = flask.Flask(__name__)
db = StrictRedis(host=REDIS_HOST, db=0)

# TODO: Save image to disk or database


@app.route("/")
def hello():
    try:
        visits = db.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html = "<h3>Hello {name}!</h3>" \
           "<b>Scooter is running.</b><br />"\
           "<b>Hostname:</b> {hostname}<br/>" \
           "<b>Visits:</b> {visits}"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname(), visits=visits)


@app.route("/predict", methods=["POST"])
def predict():
    input_data = flask.request.json['data']
    parameters = flask.request.json.get('parameters', {})
    result = _predict(input_data, parameters)
    return flask.jsonify(result)


def _predict(input_data, parameters):
    data = {"success": False}
    # TODO: Is this the place to handle missing input data?
    if not input_data:
        return data

    start_time = time.time()

    # Push job to queue
    instance_id = str(uuid.uuid4())
    element = {"id": instance_id, "x": input_data, "parameters": parameters}
    app.logger.info("Pushing job '%s' to queue", instance_id)
    db.rpush(REDIS_QUEUE, json.dumps(element))
    app.logger.info("Job pushed after %s seconds", round(time.time() - start_time, 2))

    while True:
        output = db.get(instance_id)

        if output is None:
            time.sleep(WEB_SLEEP)
            continue

        app.logger.info("Found result for job '%s'", instance_id)
        output = output.decode("utf-8")
        data["predictions"] = json.loads(output)

        app.logger.info("Removing result for job '%s'", instance_id)
        db.delete(instance_id)
        break

    data["success"] = True

    app.logger.info("Classification completed after %s seconds", round(time.time() - start_time, 2))

    return data
