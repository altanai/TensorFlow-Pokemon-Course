# TensorFlow

- end-to-end open source platform for machine learning.
- working with tensors , ehich mathematically is an algebraic object that describes a (multilinear) relationship between sets of algebraic objects related to a vector space.

It can build and train models using keras, iterate and debug . It builds Neural networks for ML . 

install tensorflow using pip 
```
pip install tensorflow
```

Libs 
```
pip install numpy
pip install pandas  
pip install matplotlib 
```

test import them in cmd python
```
> python 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
```
### Proj 1 : model to try to predict whether a Pokémon is a legendary Pokémon
using kaggle db for data miniming on pokemon - https://www.kaggle.com/alopez247/pokemon

### Proj 2 : neural network that classifies images.

step 1. Downlad handwriten digits dataset from MNIST using googleapis
ref - http://yann.lecun.com/exdb/mnist/

### Proj 3: ML is a classifier trained over the Fashion MNIST dataset

dataset compromises of 70,000 grayscale images of articles of clothing.
The greyscale values for a pixel range from 0-255 (black to white). Each low-resolution image is 28x28 pixels

```
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

2. Build the tf.keras.Sequential model by stacking layers.

**Preprocessing the dataset**
Greyscale assigned to each pixel within an image has a value range of 0-255. To flatten each image is stored as a 2-dimensional array where each numerical value in the array is the greyscale code of particular pixel. 
```
>>> train_images = train_images / 255.0
>>> test_images = test_images / 255.0
```

**Model Generation** 
Every NN is constructed from a series of connected layers that are full of connection nodes and have its own particular mathematical operation
This models uses 3 layers 
Layer 1 - take an image and format the data structure in a method acceptable for the subsequent layers such as take multidemsion and produce sinngle dimension ( flatten ) . 
Layer 2 - Dense layers with 128 node uses Rectified Linear Unit (ReLU) Activation Function that outputs values between zero and 1
Layer 3 - Dense layers with 10 node uses softmax activation function to outpupt probabilities , scales everything to add up to 1.

```
model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)), 
keras.layers.Dense(128, activation=tf.nn.relu), 
keras.layers.Dense(10, activation=tf.nn.softmax)])
```

**Training the Model**
before training define models optimizer , loss function and metrics . 
```
model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```
train the model with our Flatten array , correct classification and  number of epochs undertaken
```
model.fit(train_images, train_labels, epochs=5)
```
Observe the progress with eveery testing cycle 
```
Epoch 1/5
60000/60000 [==============================] - 3s 45us/sample - loss: 0.4995 - acc: 0.8244
Epoch 2/5
60000/60000 [==============================] - 2s 41us/sample - loss: 0.3721 - acc: 0.8654
Epoch 3/5
60000/60000 [==============================] - 3s 53us/sample - loss: 0.3348 - acc: 0.8788
Epoch 4/5
60000/60000 [==============================] - 3s 46us/sample - loss: 0.3119 - acc: 0.8866
Epoch 5/5
60000/60000 [==============================] - 4s 61us/sample - loss: 0.2925 - acc: 0.8928
```
softmax activation function of the final output layer will provide a probability that the image belongs to each of the 10 label categories.

**Evaluating the Model**
After obtaining a functional and trained NN model, evaluate model . Find test accuracy
```
test_loss, test_acc = model.evaluate(test_images, test_labels)
```
you can outout the loss and accuracy 
```
>>> test_loss
0.36592924320697784
>>> test_acc
0.8663
```

Feed test exmaples from the test array . 
```
predictions = model.predict(test_images)
```
Observer for a single result predicted , most value are closer to zero , with raised to exponenets such as e-06 . however the value 9.6431386e-01 is more close to 1 ie 96.43% hence it classofies under label with index 9
```
 predictions[0]
array([1.6079561e-05, 3.8696378e-08, 3.3632841e-07, 9.7230007e-08,
       9.6133635e-06, 1.1564302e-02, 2.6221289e-06, 2.4083924e-02,
       9.1205711e-06, 9.6431386e-01], dtype=float32)

```
Can look through a list to determine the class label , hence the prediction is label 9
```
>>> numpy.argmax(predictions[0])
9
```
Finally, we can verify this prediction by looking at the label ourselves, indeed this object had label 9 
```
>>> test_labels[0]
9
```

##  debugging 

**Issue 1**  Outdated python version 

If you have python v2 installed such as 2.7 and u get warning like 
```
DEPRECATION: Python 2.7 reached the end of its life on January 1st, 2020. Please upgrade your Python as Python 2.7 is no longer maintained. pip 21.0 will drop support for Python 2.7 in January 2021. More details about Python 2 support in pip, can be found at https://pip.pypa.io/en/latest/development/release-process/#python-2-support
```
**Solution** set python alias and update pyton 
open bash_cliases file 
```
vi ~/.bash_aliases
```
and add alias for python version 
```
alias python=python3
```
implement the changes 
```
source ~/.bash_aliases
```
recheck python version , it should be 3xx
```
python --version       
Python 3.6.9
```

**Issue2** pip outdated
**solution** as described above check pip version and make an alias 
```
> pip3 --version
pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)
➜  TensorFlow-Pokemon-Course git:(master) pip --version 
\WARNING: pip is being invoked by an old script wrapper. This will fail in a future version of pip.
Please see https://github.com/pypa/pip/issues/5599 for advice on fixing the underlying issue.
To avoid this problem you can invoke Python with '-m pip' instead of running pip directly.
pip 20.1.1 from /home/altanai/.local/lib/python2.7/site-packages/pip (python 2.7)
```
