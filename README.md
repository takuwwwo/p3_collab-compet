# Project2 Tennis

## Task 
In this task, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Total rewards of two agents are evaluated by the max value of the two rackets' rewards.

The observation space of each agent is 24 dimensions corresponding to the position and velocity of the ball and two rackets. Each agent is required to select a two-dimensional action, of which each coordinate is a number between -1 and 1.

## Dependencies
### 1. Setup the Python Environment
#### 1. Prepare Python environment with version 3.6
#### 2. Install necessary libraries
```
cd python
pip install .
```

### 2. Download the Unity Environment given by Udacity
Download the Unity environment depending on your operating system.
- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

Unzip the file and place it in this directory.

## How to Run
```
python train.py
```
You can change the settings by modifying the settings.py.
