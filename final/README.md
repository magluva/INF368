Sligthly more optimized LeNet-5 implementation.

-----------------------------------------------------------
Files:

-----------------------------------------------------------
MNIST_final_model.py
``
Training, performing cross validation and saving network to .json
``
final_model.py
``
Evaluates model
``
CV_final.png
``
Plot of K accuracy data
``
CV_LeNet_final_1.json
``
Saved best model
``
CV_LeNet_final_1_weights.h5
``
Saved weights
``
res_dict_final
``
Dictionary of cross validation training data and metrics.
``
Results_final.txt
``
Evaluation results
``

-----------------------------------------------------------

Model architecture:

Layer (type)-----------------(Output Shape)---------------Param #

conv2d_1 (Conv2D)------------(None, 26, 26, 6))---------------60

average_pooling2d_1 (Average (None, 13, 13, 6)----------------0

conv2d_2 (Conv2D)------------(None, 11, 11, 16)-------------880

average_pooling2d_2 (Average (None, 5, 5, 16)-----------------0

flatten_1 (Flatten)----------(None, 400)----------------------0

dense_1 (Dense)--------------(None, 120)-------------------48120

dense_2 (Dense)--------------(None, 84)--------------------10164

dense_3 (Dense)--------------(None, 10)----------------------850

Total params: 60,074 Trainable params: 60,074 Non-trainable params: 0
