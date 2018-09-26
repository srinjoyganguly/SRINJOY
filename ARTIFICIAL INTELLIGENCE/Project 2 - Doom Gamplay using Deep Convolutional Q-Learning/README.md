# DOOM GAMEPLAY USING DEEP CONVOLUTIONAL Q-LEARNING

![a-typical-doom-game-screen png](https://user-images.githubusercontent.com/35863175/46059427-d052e880-c17c-11e8-988d-b22377752f8c.jpg)

[Doom](https://en.wikipedia.org/wiki/Doom_(1993_video_game)) is a first person shooter, single player/multi player video game developed in 1993. In this project, we are going to implement the Doom gameplay, not with humans playing, but with our AI as an agent playing the game (environment) with it's own intelligence and developing it's own wonderful strategies/policies to win the game which sometimes humans cant even think. To do this, [OpenAI Gym](https://gym.openai.com/) provides a rich collection of reinforcement learning environments using a unified interface. It exposes several methods and fields that provide the required information about an environment's capabilities. Ww will be importing the environment from OpenAI Gym for implementing our project.

## Prerequisites for this project

To understand this advanced project, there are some prerequisites which needs to be met so that you can enjoy the benefits of this project. Let's summarize them - 
* The online course of AI from UC Berkeley, which you can find [here](http://ai.berkeley.edu/lecture_videos.html) - most important topics are - reinforcement learning, Bellman equation, Markov Decision Process, Temporal Difference, Q-Learning, Deep Q-Learning and experience replay.
* A good tutorial on Deep Convolutional Q-Learning can be found [here](https://vmayoral.github.io/robots,/ai,/deep/learning,/rl,/reinforcement/learning/2016/08/07/deep-convolutional-q-learning/) and Eligibility Trace is provided [here](https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html) and in this [link](http://pierrelucbacon.com/traces/).
* Programming experience in Python and [Pytorch library](https://pytorch.org/) which will be used in this heavily. Pytorch tutorials can be found at this [link](https://pytorch.org/tutorials/).

## Installation Instructions
### For Windows, Mac or Linux
1. Install Anaconda software - Python 3.6 version from [here](https://www.anaconda.com/download/). This software comes with the Python IDE and lots of more libraries for data science and machine learning such as Numpy, pandas, matplotlib, scipy etc.
2. Open the Anaconda Prompt in windows or terminal in Ubuntu or Mac system.
* It's always better to create a separate environement for your project (so that the project contents/settings do not interfere with your original root folder) by running this command - conda create -n env-name python=3.x anaconda. In this x is for the version for example 2.7, 3.5 or 3.6 etc and activate the environment with the commands shown in the terminal after you make the environment.
* For this project create this environment - conda create -n aidoom python=3.5 anaconda. You can change the environment as you like.
3. Install Pytorch library using the instructions given [here](https://pytorch.org/get-started/locally/). It is recommended to install the No CUDA - CPU version of Pytorch if you do not have the proper Nvidia graphics card drivers installed or do not have any graphics card. If you have the proper graphics card and it's driver then you are welcome to install the CUDA versions of Pytorch according to your version. To install previous versions of Pytorch click [here](https://pytorch.org/get-started/previous-versions/) and install using pip instructions. Pip ensures that you install everything in your environment only and do not interferes with your root system, however if you install using conda, then it is installed for all the enviornemnts and this interferes with all other environments in your system.
4. Now follow these steps in sequence and run them one by one in your terminal or Anaconda prompt - 
* sudo apt-get install  build-essential
* sudo apt-get install emacs
* sudo apt-get update
* sudo apt-get install python3.5
* sudo apt-get install python3-pip
* sudo apt install git
* sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
* pip install ppaquette-gym-doom
* pip install gym==0.7.4
* pip3 install gym-pull
* sudo  apt-get install -y python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip git
* conda install -c conda-forge ffmpeg
* pip3 install pyopengl
* sudo apt-get install libstdc++6
* sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
* sudo apt-get update
* sudo apt-get upgrade
* sudo apt-get dist-upgrade
* conda install libgcc
* cd doom-py
* python setup.py install
* pip install -e
* pip install domm_py == 0.0.15
5. In case of any error, please consult the debugging files provided with the project. A common error is - TypeError: multinomial() missing 1 required positional arguments: "num_samples". In this case downgrading the pytorch version will help. The verison must be less than 0.4.0.

## Code Snippets with Detailed Explanation
### Elucidation of ai.py file
![importing libraries](https://user-images.githubusercontent.com/35863175/46059437-de086e00-c17c-11e8-8b3e-c76b9f304312.JPG)

![class cnn](https://user-images.githubusercontent.com/35863175/46059450-e791d600-c17c-11e8-8399-e5de5da1d1aa.JPG)

![body](https://user-images.githubusercontent.com/35863175/46059467-f37d9800-c17c-11e8-8055-31e570b9d21f.JPG)

![class ai](https://user-images.githubusercontent.com/35863175/46059470-f8dae280-c17c-11e8-86a1-3b89cc634858.JPG)

![getting doom building ai and setting experience replay](https://user-images.githubusercontent.com/35863175/46059476-02644a80-c17d-11e8-9aac-4d0806ac55e3.JPG)

![eligibility trace](https://user-images.githubusercontent.com/35863175/46059486-0bedb280-c17d-11e8-9e87-a72e03522646.JPG)

![moving average on 100 steps](https://user-images.githubusercontent.com/35863175/46059490-10b26680-c17d-11e8-8e2b-87c46024db3d.JPG)

![training the ai](https://user-images.githubusercontent.com/35863175/46059497-190aa180-c17d-11e8-986c-941458cf47c2.JPG)



## Running our Doom Gameplay
You can see the outputs in the videos folder provided with this project folder.

## Acknowledgements
* Udemy online platform for sustaining this beautiful course on AI.
* Huge ton of thanks to Hadelin De Ponteves and Kirill Ermenko for creating this wonderful course on AI
* Lots of thanks to Zenodia Charpy for providing the appropriate instructions of OpenAI Gym on Windows, and those instructions also worked for me in my Ubuntu 16.04.
* Style of explanation of the code is inspired from Adrian Rosebrock. His [Linkedin](https://www.linkedin.com/in/adrian-rosebrock-59b8732a) and [website](https://www.pyimagesearch.com/author/adrian/)
* Google Images.












