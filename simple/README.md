Simple model with two fullt connected (Dense) layers.

Files:
MNIST_simple.py
  Trains model and saves it and the weights.
  
simple.py
  Loads and evaluates model.
 
MNIST_simple.json
  The saved model.
  
MNIST_simple.h5
  The saved weights.

MNIST_simple.png
  Training data plot
 
Results_simple.txt
  Results (confusion matrix etc.)

-----------------------------------------------------------------
Model architecture:

-----------------------------------------------------------------

Layer (type)---------------Output Shape---------------Param # 

-----------------------------------------------------------------

dense_1 (Dense)-------------(None, 32)---------------25120  

-----------------------------------------------------------------

dense_2 (Dense)-------------(None, 10)---------------330      

-----------------------------------------------------------------

Total params: 25,450
Trainable params: 25,450
Non-trainable params: 0

-----------------------------------------------------------------
