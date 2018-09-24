# SELF DRIVING CAR SIMULATION

In this project we are going to simulate a self driving car in a Kivy environment. We will be training the brain of the car using Deep Q learning AI algorithm and will be drawing the map on the Kivy map environment for our car to navigate in that environment. We will be testing various maps for checking the performance of our car.

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
4. Download https://kivy.org/downloads/appveyor/kivy/Kivy-1.10.1.dev0-cp36-cp36m-win_amd64.whl
5. Go to the download folder in your Anaconda Prompt console ("cd folder name"), the folder where you downloaded the file from step 4.
6. Enter the following commands in the Anaconda Prompt console to install Kivy:-
* pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew
* pip install kivy.deps.gstreamer
* pip install Kivy-1.10.1.dev0-cp36-cp36m-win_amd64.whl
7. In Spyder go to Tools -> Preferences -> Run
8. Set Console -> Execute in an external system terminal

### For Mac or Linux
1. Open your terminal.
2. Install Anaconda using the instructions provided [here](https://conda.io/docs/user-guide/install/linux.html) or you cna follow these [instructions](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04).
* Here also create a separate environment using the command provided in the windows installation instructions. After installing anaconda, type that command and and activate the environment with the commands shown in the terminal after you make the environment.
3. Install Pytorch using the link already given in the windows installation instructions. There you can select your operating system, CUDA etc and run the commands provided there.
4. Install Kivy using the instructions given in this [link](https://kivy.org/doc/stable/installation/installation-linux.html) for Linux based systems. Choose the command for stable builds.
5. Install Kivy using the instructions given in this [link]https://kivy.org/doc/stable/installation/installation-osx.html() for MacOSX based systems. You can also use the Homebrew with pip in this.

## Code Snippets with Detailed Explanation
### Elucidation of ai.py file
![importing libraries](https://user-images.githubusercontent.com/35863175/45941623-bb4f4b80-bffc-11e8-835f-98283cc8557f.JPG)
* [Numpy](http://www.numpy.org/) is imported in line 5 and is used for the scientific computation and linear algebra. In line 6, [random library](https://www.pythonforbeginners.com/random/how-to-use-the-random-module-in-python) is imported which is used for picking random elements from a list or from different batches, [os](https://www.pythonforbeginners.com/os/pythons-os-module) in line 7 is for using the operating system based functions and for interfacing python with OS such as Windows, Mac or  linux. From lines 8 to 11, we import all [torch modules](https://pytorch.org/docs/stable/nn.html) for creating our neural network, calculating loss functions and optimizers. Lines 12 and 13 is for conversion of torch tensors into variables which contain gradients.

![architecture of nn](https://user-images.githubusercontent.com/35863175/45941635-ca35fe00-bffc-11e8-9251-041f74648fff.JPG)

* This is the creation of the architecture of our neural network. 
* In line 17, class is created which inherits tools of the module class to make our neural network. This is called [inheritance](https://www.programiz.com/python-programming/inheritance) which is an imporatnt concept in object oriented programming. 
* In line 19, init function is created which is always done in python to initialize our class object (self) and this takes three arguments - self, input_size refers to 3 signals + orientation where orientation is a 5D encoded vector (neuron number for input) and nb_action which are the number of output neurons. 
* Line 20 is used to activate the inheritance. 
* Lines 21 and 22 are for initializing the inputs and ouputs of the NN. 
* Line 23 is full connection between the input neurons layer and a hidden layer with 30 neurons.
* Line 24 is full connection between hidden layer and the output layer (nb_action).
* In Line 26 forward function is defined with our class object self and state which is the input entering our NN.
* From lines 27 to 29, We activate the hidden layer with [Relu](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) and return the output as q values.

![replaymemory](https://user-images.githubusercontent.com/35863175/45941655-df129180-bffc-11e8-8b4e-eb9d9e1218da.JPG)
* Replay memory.

![dqn_init_select_action](https://user-images.githubusercontent.com/35863175/45941667-edf94400-bffc-11e8-9346-35f78a469fb6.JPG)
* Deep Q Network (Dqn) class is created which has various functions to implement the Deep Q-Learning AI algorithm and will form the brain of our self driving car. Since this class is quite large with lots of functions inside it, we have divided it into chunks for better explanation and understanding.

![learn and update](https://user-images.githubusercontent.com/35863175/45941675-f5b8e880-bffc-11e8-9b18-84b4b9570c0c.JPG)
* Learning and updation of weights of the self driving car network.

![score save and load](https://user-images.githubusercontent.com/35863175/45941706-05383180-bffd-11e8-9db8-e1826d1152da.JPG)
* Scores will be assigned to the car according to it's performance. If it hits the wall, negative scores wil be provided and when it goes smoothly without any hit, it will generate more positive scores. 

## Acknowledgements
* Udemy online platform for sustaining this beautiful course on AI.
* Huge ton of thanks to Hadelin De Ponteves and Kirill Ermenko for creating this wonderful course on AI
* Lots of thanks to Christian Rosan for providing the appropriate instructions of Kivy installation on Windows.
* Style of explanation of the code is inspired from Adrian Rosebrock. His [Linkedin](https://www.linkedin.com/in/adrian-rosebrock-59b8732a) and [website](https://www.pyimagesearch.com/author/adrian/)













