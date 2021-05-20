% Modern Deep Learning Libraries: Jax, Trax and Others
% M. Yusuf Sarıgöz
% 02/27/2021

---

# Jax

---

## What is it?
::: Definition
* Drop-in replacement for Numpy: `import jax.numpy as jnp`.
* Accelerated on hardware with `XLA`.
* Automatically differentiable with `Autograd`.
* Brought together for high performance ML research.
:::
::: Reference
* Some of the following code samples are inspired from Mat Kelcey's blog at [matpalm.com/blog/ymxb_pod_slice](http://matpalm.com/blog/ymxb_pod_slice/)
:::

---

## Tracing functions
```python
def myfunc(x):
  return 2*x + 3

myfunc(5)

from jax import make_jaxpr
trace_myfunc = make_jaxpr(myfunc)
trace_myfunc(5)
```

---

## Differentiating functions
```python
from jax import grad, value_and_grad

grad_myfunc = grad(myfunc)
grad_myfunc(5.)

value_grad_myfunc = value_and_grad(myfunc)
value, gradient = value_grad_myfunc(5.)
```

---

## Jitting functions
```python
from jax import jit

jitted_myfunc = jit(myfunc)
jitted_myfunc(5)

trace_jitted_myfunc = make_jaxpr(jitted_myfunc)
trace_jitted_myfunc(5)
```
::: Notes
* You can also decorate functions with `@jax.jit`.
:::

---

## Vectorizing functions
```python
from jax import vmap

def dense(x, w, b):
  return jnp.dot(x, w) + b

vectorized_dense = vmap(dense, in_axes=(0, None, None))

jitted_vectorized_dense = jit(vmap(dense, in_axes=(0, None, None)))
```

## Parallelizing functions
```python
from jax import pmap

parallelized_dense = pmap(dense, in_axes=(0, None, None))

jitted_parallelized_dense = jit(pmap(dense, in_axes=(0, None, None)))
```

## Why Jax?
::: Benefits
* No gradient tapes or graphs. Simply `grad` it.
* Don't be bothered by the batch dim: write for single sample and then `vmap` it.
* Just `jit` it for easy hardware acceleration.
* Parallelize your computation and `pmap` it to devices.
:::
::: Modules
* `jax.numpy.fft`
* `jax.scipy.signal`
* `jax.numpy.hamming`
:::

## Higher-level libs
* Stax 
* *Flax*
* Haiku
* Objax
* *Elegy*
* *Trax*

# Flax
## What is it?
::: Pros
* Clean and flexible coding.
* Useful error messages.
* Easy expandability and extendability.
:::
::: Cons
* Immature and hacky.
* Dependence on TensorFlow for data pipelines.
:::

## Sample
```python
class CNN(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x
```

# Elegy
## What is it?
* Keras-like high-level APIs for productivity.
* Functional PyTorch Lightning-like low-level APIs for flexibility.
* 100% framework-agnostic trainer APIs.
* Supports Flax, Haiku, Optax and many others.
* Consumes many data sources including TensorFlow datasets, PyTorch data loaders, Python generators and Numpy pytrees.

## Sample
```python
model = elegy.Model(
    module=CNN(),
    loss=[
        elegy.losses.SparseCategoricalCrossentropy(from_logits=True),
        elegy.regularizers.GlobalL2(l=1e-5),
    ],
    metrics=elegy.metrics.SparseCategoricalAccuracy(),
    optimizer=optax.rmsprop(1e-3),
)

model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    steps_per_epoch=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[elegy.callbacks.TensorBoard("summaries")]
)
```

# Trax
## What is it?
::: Description
* End-to end machine learning library with a focus on clear code and speed.
* Actively used and maintained by Google Brain, the team behind the many recent innovations in NLP.
:::
::: Visit
[github.com/google/trax](https://github.com/google/trax)
:::

## Why Trax?
::: Reasons
* Quick access to `tensorflow-datasets` with `trax.data.TFDS`.
* Easy NLP with built-in `trax.data.tokenize`, trax.supervised.decoding.autoregressive_sample` and `trax.data.detokenize` utils.
* Configurable SOTA models in `trax.models`.
* Stable yet simple data feeding API with Python generators.
:::
::: Samples
[github.com/monatis/trax-samples](https://github.com/monatis/trax-samples)
:::



## Thank you
* Contact me at [yusufsarigoz@gmail.com](mailto:yusufsarigoz@gmail.com)