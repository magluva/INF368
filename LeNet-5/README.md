Unoptimized LeNet-5 implementations

-----------------------------------------------------------------
Files:

-----------------------------------------------------------------
MNIST_LeNet.py
``
Trains model and saves it and the weights.
 ``
LeNet.py
 ``
Loads and evaluates model.
  ``
MNIST_LeNet.json
 ``
The saved model.
 ``
MNIST_LeNet.h5
 ``
The saved weights.
 ``
MNIST_LeNet.png
 ``
Training data plot
  ``
lenet_train_hist_dict
``
Saved training data
``
Results_LeNet.txt
 ``
Results (confusion matrix etc.)
 ``
 Comparison.py
 ``
 compares this model with the simple model
 ``
 comparison.png
 ``
 Comparison plot between LeNet and simple 
 ``

Model architecture:

-----------------------------------------------------------------
Layer (type)-----------------(Output Shape)---------------Param #   

-----------------------------------------------------------------
conv2d_1 (Conv2D)------------(None, 26, 26, 6))---------------60        

-----------------------------------------------------------------
average_pooling2d_1 (Average (None, 13, 13, 6)----------------0       

-----------------------------------------------------------------
conv2d_2 (Conv2D)------------(None, 11, 11, 16)-------------880     

-----------------------------------------------------------------
average_pooling2d_2 (Average (None, 5, 5, 16)-----------------0         

-----------------------------------------------------------------
flatten_1 (Flatten)----------(None, 400)----------------------0      

-----------------------------------------------------------------
dense_1 (Dense)--------------(None, 120)-------------------48120   

-----------------------------------------------------------------
dense_2 (Dense)--------------(None, 84)--------------------10164     

-----------------------------------------------------------------
dense_3 (Dense)--------------(None, 10)----------------------850   

-----------------------------------------------------------------
Total params: 60,074
Trainable params: 60,074
Non-trainable params: 0

-----------------------------------------------------------------
