"""Scooter Constants Default Values."""
import os

# Redis host name
REDIS_HOST = os.environ.get('SCOOTER_REDIS_HOST', 'localhost')

# Name of the list used as queue on Redis
REDIS_QUEUE = os.environ.get('SCOOTER_REDIS_QUEUE', 'prediction:queue')

# The amount of time the web threads sleep before looking for completed prediction.
WEB_SLEEP = float(os.environ.get('SCOOTER_WEB_SLEEP', "0.10"))

# The amount of time the worker sleeps before looking for more work.
WORKER_SLEEP = float(os.environ.get('SCOOTER_WORKER_SLEEP', "0.05"))

# Maximum batch size used when making predictions.
BATCH_SIZE = int(os.environ.get('SCOOTER_BATCH_SIZE', "16"))
