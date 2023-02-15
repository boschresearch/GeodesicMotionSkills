Geodesic Motion Generator
=========================

This repository constains the code for the paper *Learning Riemannian Manifolds for Geodesic Motion Skills* submitted at Robotics: Science and Systems 2021 conference. 

**Warning**: The development of this framework is ongoing, and thus some substantial changes might occur. Sorry for the inconvenience.

Requirements
------------
The code has been tested with Python 3.8 on Ubuntu 20.04. The installation on other OS is experimental.

Installation
------------
1. First download the ``pip`` Python package manager and create a virtual environment for Python as described in the following link: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
On Ubuntu, you can install ``pip`` and ``virtualenv`` by typing in the terminal: 

- In Python 3.8:

```
	sudo apt install python3-pip
	sudo pip install virtualenv
```
If you have them installed then skip this step

2. Clone the repository and install the requirements as follows

```
mkdir gmg_repo
cd gmg_repo

git clone -b main https://<personal-token>@github.boschdevcloud.com/HAB2RNG/GeodesicMotionGenerator.git

# If your Github personal token is not activated use the following command instead 
git clone -b main https://github.boschdevcloud.com/HAB2RNG/GeodesicMotionGenerator.git

# Clone Stochman from here (in the same directory)
git clone https://github.com/MachineLearningLifeScience/stochman
cd stochman/
git checkout 1d092e0dffef179b706542f0693b813a8370fcf8
cd ..

# Clone hyperspherical vae from here (in the same directory)
git clone https://github.com/hadibeikm/s-vae-pytorch.git


cd GeodesicMotionGenerator

# Create a virtual environment and activate it
virtualenv -p /usr/bin/python3.8 gmg
source gmg/bin/activate

# Install the requirements
pip install -r requirements.txt

# Navigate to s-vae-pytorch directory
cd ../s-vae-pytorch
# Install the package
pip install -e .

# Navigate stochman directory
cd ../stochman
# Install the package
pip install -e .

# Add the repo directory to your python path (/your/repo/directory/ is where you clone the repos)
export PYTHONPATH="${PYTHONPATH}:/your/repo/directory/gmg_repo"
```

3. Execute the example code
```
# Navigate to the Experiments folder in GeodesicMotionGenerator directory
cd ../GeodesicMotionGenerator/Experiments/
python toy_example.py 
# This command will generate metric plot for our toy example mentioned in the paper
```
4. You can use Pycharm to open the project and change the hyperparamters for training and test. 
