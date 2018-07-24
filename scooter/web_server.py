"""Scooter RESTful API."""

import logging
import json
import os
import socket
import time
import uuid

import flask
from redis import RedisError, StrictRedis

# initialize constants used for server queuing
# TODO: Deduplicate these
# TODO: We could load them from environment variables if nothing else works.
PREDICTION_QUEUE = "prediction:queue"
CLIENT_SLEEP = 0.25

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
db = StrictRedis(host=os.environ['SCOOTER_REDIS'], db=0)

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
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    input_data = flask.request.json['data']
    if not input_data:
        return data

    parameters = flask.request.json.get('parameters', {})

    # generate an ID for the prediction then add the ID + data to the queue
    x_id = str(uuid.uuid4())
    element = {"id": x_id, "x": input_data, "parameters": parameters}
    app.logger.info("Pushing job '%s' to queue", x_id)
    db.rpush(PREDICTION_QUEUE, json.dumps(element))

    # keep looping until our model server returns the output predictions
    while True:
        # attempt to grab the output predictions
        output = db.get(x_id)

        # check to see if our model has classified the input image
        if output is None:
            time.sleep(CLIENT_SLEEP)
            continue

        logging.info("Found result for job '%s'", x_id)
        # add the output predictions to our data dictionary so we can return it to the client
        output = output.decode("utf-8")
        data["predictions"] = json.loads(output)

        # delete the result from the database and break from the polling loop
        logging.info("Removing result for job '%s'", x_id)
        db.delete(x_id)
        break

    # indicate that the request was a success
    data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)
