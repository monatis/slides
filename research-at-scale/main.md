% Research at Scale with TFRC and Cloud TPUs
% M. Yusuf Sarıgöz
% 05/20/2021

## TFRC
- Apply for access to a cluster of TPUs for 30 days.
- Use it for your research, open-source projects, content creation and similar.
- Publish your results with peer-reviewed papers, open-source code, blog posts and similar.
- Share your feedback with Google.
- Reapply if you need 30 days more.
- [tensorflow.org/tfrc](https://www.tensorflow.org/tfrc)

## My Work with TFRC
- Contributed to [github.com/TensorSpeech/TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR) and [github.com/tensorspeech/TensorFlowTTS](https://github.com/tensorspeech/TensorFlowTTS).
- Trained neural TTS models in German and published them at [tfhub.dev/monatis](https://tfhub.dev/monatis).
- Created an "AI as a service" project with pretrained production-ready NLP models: [github.com/monatis/ai-aas](https://github.com/monatis-aas)
- Writing a research paper on emotional TTS in German.
- Writing a research paper on multitasking NLP model in Turkish.
- Reapplied for 30 days more in order to train ASR models in multiple languages.

## Simple TPU Initialization
```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='<Colab TPU address or Google Compute Engine TPU name goes here>')
tf.config.connect_to_cluster(resolver)
tf.tpu.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
```

## Easy Distribution with TPU Strategy
```python
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = ...
```

## Tips for Effective TPU
- Use static shapes for batches and sequences.
- Preprocess your data as much as possible and store them in TFRecords.
- Subclass `tf.keras.layers.Layer` for your innovations.
- Use custom model.fit() when necessary.
- Enjoy new `tensorflow-cloud` Python package for fastest route from notebook to the cloud.

## Custom Layer
```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

## Automatic Tracking of Weights
```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

## Non-trainable Weights
```python
class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total
```

## Lazy Initialization
```python

class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

## Model Class
The Model class has the same API as Layer, with the following differences:

- It exposes built-in training, evaluation, and prediction loops ( model.fit() , model.evaluate() , model.predict() ). 
- It exposes the list of its inner layers, via the model.layers  property. 
- It exposes saving and serialization APIs ( save() , save_weights() ...) 

## Custom model.fit()
- Create a new class that subclasses `keras.Model`.
- Override the method `train_step(self, data)`.
- Return a dictionary mapping metric names (including the loss) to their current value.
- The input argument `data` of `train_step` is what gets passed to fit as training data:
- If you pass `Numpy` arrays, by calling `fit(x, y, ...)`, then `data` will be the tuple `(x, y)`.
- If you pass a `tf.data.Dataset`, by calling `fit(dataset, ...)`, then `data` will be what gets yielded by `dataset` at each batch.

## Custom Model Class
```python
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
```

## How to Use Custom Model
```python
import numpy as np

# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Just use `fit` as usual
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)
```

## tensorflow-cloud
- Develop locally and train in the cloud with a single function call.
- Removes the headache of packaging your research code as well as creating and managing cloud instances.
- Robust, predictable, flexible and reproduceable as it is completely Docker-based.
- [bit.ly/tfc-keras](http://bit.ly/tfc-keras)

```python
tfc.run(
    docker_image_bucket_name=gcp_bucket,
    chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
    worker_count=1,
    worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"]
)
```

## Thank you
- [yusufsarigoz@gmail.com](mailto:yusufsarigoz@gmail.com)
- [bit.ly/mys-linkedin](http://bit.ly/mys-linkedin)
- [github.com/monatis](https://github.com/monatis)
