# DOOM GAMEPLAY USING DEEP CONVOLUTIONAL Q-LEARNING









## Prerequisites for this project

To understand this advanced project, there are some prerequisites which needs to be met so that you can enjoy the benefits of this project. Let's summarize them - 
* The online course of AI from UC Berkeley, which you can find [here](http://ai.berkeley.edu/lecture_videos.html) - most important topics are - reinforcement learning, Bellman equation, Markov Decision Process, Q-Learning, Deep Q-Learning and experience replay.
* Programming experience in Python and Pytorch library which will be used in this heavily.

## Installation Instructions
### For Windows
1. Install Anaconda software - Python 3.6 version from [here](https://www.anaconda.com/download/). This software comes with the Python IDE and lots of more libraries for data science and machine learning such as Numpy, pandas, matplotlib, scipy etc.
2. Open the Anaconda Prompt, which acts like a terminal in windows environment similar to an Ubuntu or Mac system.
* It's always better to create a separate environement for your project (so that the project contents/settings do not interfere with your original root folder) by running this command - conda create -n env-name python=3.x anaconda. In this x is for the version for example 2.7, 3.5 or 3.6 etc and activate the environment with the commands shown in the terminal after you make the environment.
3. Install Pytorch library using the instructions given [here](https://pytorch.org/get-started/locally/). It is recommended to install the No CUDA - CPU version of Pytorch if you do not have the proper Nvidia graphics card drivers installed or do not have any graphics card. If you have the proper graphics card and it's driver then you are welcome to install the CUDA versions of Pytorch according to your version. To install previous versions of Pytorch click [here](https://pytorch.org/get-started/previous-versions/) and install using pip instructions. Pip ensures that you install everything in your environment only and do not interferes with your root system, however if you install using conda, then it is installed for all the enviornemnts and this interferes with all other environments in your system.
