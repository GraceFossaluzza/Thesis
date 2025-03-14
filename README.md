# Authors
Grace Fossaluzza, grace.fossaluzza@mail.polimi.it


# Introduction
This repo contains the code used for the paper [Time series data estimation using Neural ODE in Variational Auto Encoders]().

Using pytorch and Neural ODEs (NODEs) it attempts to learn the true dynamics of time series data using 
toy examples such as clockwise and counterclockwise spirals, and three different examples of sine waves: 
first a standard non-dampened sine wave, second a dampened sine wave, third an exponentially decaying and 
dampened sine wave. Finally, the NODE is trained on real world time series data of solar power curves.

The performance of the NODEs are compared to an LSTM VAE baseline on RMSE error and time per epoch.  

This project is a purely research and curiosity based project.

# Code structure
To make development and research more seamless, an object-oriented approach was taken to improve efficiency and
consistency across multiple runs. This also makes it easier to extend and change workflows across multiple models at once.

## Source files
The src folder contains the source code. The main components of the source code are:

- `data.py`: Data loading object. Primarily uses data generation functions.
- `model.py`: Contains model implementations and the abstract `TrainerModel` class which defines models
in the `trainer.py` file.
- `train.py`: A generalized `Trainer` class used to train subclasses of the `TrainerModel` class.
Moreover, it saves and loads different types of models and handles model visualizations.
- `utils.py`: Standard utility functions
- `visualize.py`: Visualizes model properties such as reconstructions, loss curves and original data samples


Each `main.py` script takes a number of relevant parameters as input to enable parameter tuning,
experimentation of different model types, dataset sizes and types. These can be read from the respective files.

# Running the code
To run the code use the following code in a terminal with the project root as working directory:
`python -m src.[dataset].main [--args]`

For example:
`python3 -m src.toy.main --epochs 1000 --freq 100 --latent-dim 4 --hidden-dim 30 --lstm-hidden-dim 45 --lstm-layers 2 --lr 0.001 --solver rk4`

## Setup environment
Create a new python environment and install the packages from `requirements.txt` using

`pip install -r requirements.txt`

## Run python notebook
Install Jupyter with `pip install jupyter` and run a server using `jupyter notebook` or any supported software
such as Anaconda.
