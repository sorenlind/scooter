Scooter Examples
================

These are simple Python scripts showing how you can build APIs for common classifiers using Scooter.

To run the example, first make sure you have Scooter installed as well as having Redis installed and running as
explained in the Scooter README.

ResNet50
--------

This example exposes an image classifier using the [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50) model with Keras and TensorFlow as backend.

First, install packages specific to the example:

```bash
pip install tensorflow
pip install keras
pip install Pillow
```

Then, from the `examples` folder, start the example:

```bash
python resnet.py
```

Scooter is now ready to accept POST requests. The file `puffer.json` contains a base64 encoded photo of a yellow puffer
fish which you can use to test your API.

```bash
curl -H "Content-Type: application/json" -X POST --data @puffer.json 'http://localhost:5000/predict'
```

This should return the following:

```json
{"predictions":[{"label":"puffer","probability":0.9764861464500427},{"label":"rock_beauty","probability":0.023036744445562363},{"label":"eel","probability":0.00012930750381201506},{"label":"lionfish","probability":0.00012540673196781427},{"label":"coral_reef","probability":5.917380258324556e-05}],"success":true}
```

As you can see, the model as correctly identified the puffer 🐡.