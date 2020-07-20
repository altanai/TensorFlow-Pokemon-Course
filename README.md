# TensorFlow

end-to-end open source platform for machine learning.
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
Proj 1 : model to try to predict whether a Pokémon is a legendary Pokémon
using kaggle db for data miniming on pokemon - https://www.kaggle.com/alopez247/pokemon

Proj 2 : neural network that classifies images.

1. Downlad handwriten digits dataset from MNIST using googleapis
ref - http://yann.lecun.com/exdb/mnist/

2. Build the tf.keras.Sequential model by stacking layers.


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
as described above check pip version and make an alias 
```
> pip3 --version
pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)
➜  TensorFlow-Pokemon-Course git:(master) pip --version 
\WARNING: pip is being invoked by an old script wrapper. This will fail in a future version of pip.
Please see https://github.com/pypa/pip/issues/5599 for advice on fixing the underlying issue.
To avoid this problem you can invoke Python with '-m pip' instead of running pip directly.
pip 20.1.1 from /home/altanai/.local/lib/python2.7/site-packages/pip (python 2.7)
```
