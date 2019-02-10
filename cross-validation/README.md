Unoptimized LeNet-5 implementation

-----------------------------------------------------------------
Files:

-----------------------------------------------------------------
MNIST_LeNet.py
``
Trains model and saves it and the weights.
``
CV_LeNet_model_assessment.py
``
Stratified k-fold cross validation on the model.
``
CV_LeNet_finetune_solver.py
``
Stratified k-fold cross validation on optimizer.
``
CV_LeNet__regularization_finetune_activation.py
``
Stratified k-fold cross-validation on activation functions with l2 penalty on weights.
``
K1-K2_sigmoid_ReLu.png
``
Plot of cross validation accuracies on activation functions.
``
K1-K3_SGD_acc.png
``
Plot of cross-validation accuracies on optimizer
``
Results_CV.txt
``
Results from cross validation
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
