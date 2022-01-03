# Welcome to the JAX Repository


![](/images/jax_logo_250px.png)



JAX is a combination of Automatic Differentiation and XLA (Accelerated Linear ALgebra). XLA is a compiler developed by Google to work on TPU units. **Autograd** and **XLA** are brought together by JAX to accelerate ML research by enabling high-speed numerical computing. JAX's highlights include Python-Numpy code base, ease of Differentiating complex functions, Vectorization, Parallelzation, and Just-In-Time (jit) compilation.

JAX is basically a python library, just like Tensorflow and Pytorch but a bit different, I would not compare the three as is, however this should get you more comfortable around learning JAX. There is another library called FLAX (from Google Research team) which is built on top of it, and you can draw parallels to Keras being built on top of Tensorflow.

JAX Loves Numpy. Its API is very much like Numpy. It work on all accelerators - CPU, GPU, and TPUs.

Like numpy, JAX has arrays too, but two things, one they are called instead **Device Arrays** and two, they are **Immutable**. ALso, JAX is not stateful, this would imply that if you are generating random numbers you will have to pass the random state explicitly and not implicitly unlike it happen in Python Numpy. Device Arrays means that these arrays lie on your devices - CPU, GPU, and TPU. These devices are also called **accelerators**.

JAX has something called Asynchronus Dispatch System in the background. What it does is that it immedaitely delegates a computation task on device arrays to the Accelerator as soon as it is typed in, it doesn't even wait for it to excute. This is tricky as well as very useful feature which we need to keep in mind while programming using JAX. One solution is to use **block_until_ready()** while assiging an output of a calculation to a variable.

JAX uses XLA to optimize your ML code using cool compiler tricks and the programmer doesn't have to worry about it. When you cast your function as Just-in-time or to say when you **Jit** your functions they at times become order of magnitude faster, thus giving you performance benefits.

Automatic differentiation makes JAX the most exciting library for deep learning practitioners. Derivatives are integral part of back propagation which in turn is integral to the learning process of neural networks. This is done using **grad()** fnction in JAX.

**Please refer the following Colab Notebooks for detailed explanations and Code on my github repsitory here -** 

* Introducing JAX [Jupyter Notebook here](https://github.com/yvrjsharma/JAX/blob/main/JAX_1.ipynb)

* Stateless JAX, PyTreees, and Multi Layer Perceptron [Jupyter Notebook here](https://github.com/yvrjsharma/JAX/blob/main/JAX_2.ipynb)

* JAX: PMap for parallelism and Advanced Autodiff [Jupyter Notebook here](https://github.com/yvrjsharma/JAX/blob/main/JAX_3.ipynb)
