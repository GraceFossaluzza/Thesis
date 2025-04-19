# Authors
Grace Fossaluzza, grace.fossaluzza@mail.polimi.it


# Introduction
This repo contains the code used for my Thesis work.
Using pytorch and Neural ODEs (NODEs) it attempts to learn the true dynamics of time series data (EEG signals).

The performance of the NODEs are compared to an LSTM VAE baseline on RMSE error and time per epoch.  

# Code structure
To make development and research more seamless, an object-oriented approach was taken to improve efficiency and
consistency across multiple runs. This also makes it easier to extend and change workflows across multiple models at once.

## Source files
The folder contains the source code. The main components of the source code are:

- `data.py`: Data loading object. 
- `model.py`: Contains model implementations and the abstract `TrainerModel` class which defines models
in the `trainer.py` file.
- `train.py`: A generalized `Trainer` class used to train subclasses of the `TrainerModel` class.
Moreover, it saves and loads different types of models and handles model visualizations.
- `utils.py`: Standard utility functions
- `visualize.py`: Visualizes model properties such as reconstructions, loss curves and original data samples


Each `main.py` script takes a number of relevant parameters as input to enable parameter tuning,
experimentation of different model types, dataset sizes and types. These can be read from the respective files.

## Setup environment
Create a new python environment and install the packages from `requirements.txt` using

`pip install -r requirements.txt`

## Run python notebook
Install Jupyter with `pip install jupyter` and run a server using `jupyter notebook` or any supported software
such as Anaconda.
