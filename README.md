# NB!
The file structure is a mess and a lot of duplicate code between files.
Some small adjustment to function return parameters will allow me to standardize output and thus
"save file" and "plotting" functions etc. in separate files.

# Deep Learning

Implementations of a simple neural network and the LeNet-5 acrchitecture with k-fold cross validation  

## Getting Started

Download repository and run either MNIST_simple.py or NMIST_lenet.py

### Prerequisites

Python2.7 < Needed_version < Python3.7

```
Packages:
nympy
scipy
keras
tensorflow
matplotlib

```

### Installing

Install a working python version as outlined above, then create a viritual env and install the 
needed packages.

#### Windows

Open cmd and create a viritual env with a specific python version.
Tensorflow currently supports pyton3.6.x, so I used python3.6.5 

```
virtualenv -p C:\Users\Username\AppData\Local\Programs\Python\Python36\python.exe your_env_name
```

Activate the newly created environment from cmd: C:\Users\Username>

```
your_env_name\Scripts\Activate.bat
```

Now install the packages mentioned above.
You can install one by one or all together if you want. Here I did three and three just
in case something were to go wrong.

```
pip3 install numpy scipy matplotlib
pip3 install keras tensorflow sklearn
```

## Built With

* [Keras](https://keras.io/) - The Macine Learning framework used
* [Tensorflow](https://www.tensorflow.org/api_docs/) - Backend

## Authors

* **MagLuVa** 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Under development...
