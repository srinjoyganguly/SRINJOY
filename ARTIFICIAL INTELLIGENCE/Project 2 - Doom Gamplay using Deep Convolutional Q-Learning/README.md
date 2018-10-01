# DOOM GAMEPLAY USING DEEP CONVOLUTIONAL Q-LEARNING

![a-typical-doom-game-screen png](https://user-images.githubusercontent.com/35863175/46059427-d052e880-c17c-11e8-988d-b22377752f8c.jpg)

[Doom](https://en.wikipedia.org/wiki/Doom_(1993_video_game)) is a first person shooter, single player/multi player video game developed in 1993. In this project, we are going to implement the Doom gameplay, not with humans playing, but with our AI as an agent playing the game (environment) with it's own intelligence and developing it's own wonderful strategies/policies to win the game which sometimes humans cant even think. To do this, [OpenAI Gym](https://gym.openai.com/) provides a rich collection of reinforcement learning environments using a unified interface. It exposes several methods and fields that provide the required information about an environment's capabilities. Ww will be importing the environment from OpenAI Gym for implementing our project.

## Prerequisites for this project

To understand this advanced project, there are some prerequisites which needs to be met so that you can enjoy the benefits of this project. Let's summarize them - 
* Understanding of [Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) and [Deep Learning](https://www.deeplearningbook.org/).
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
* [Numpy](http://www.numpy.org/) is imported in line 6 and is used for the scientific computation and linear algebra. 
* From lines 7 to 10, we import all [torch modules](https://pytorch.org/docs/stable/nn.html) for creating our neural network, calculating loss functions and optimizers. 
* Lines 11 is for conversion of torch tensors into variables which contain gradients.
* Lines 14 to 16 is for importing the OpenAI Gym environement and the Doom game.
* Line 19 is for importing the image_processing.py and experience_replay.py files.

![class cnn](https://user-images.githubusercontent.com/35863175/46059450-e791d600-c17c-11e8-8399-e5de5da1d1aa.JPG)
* In line 27, class CNN is initialized using inheritance from nn.Module from pytorch and in line 28, init function is defined with the super method used for activating the inheritance.
* From lines 30 to 32, convolutional layers are defined for the CNN where in_channels is the input, out_channels (number of features) is the output of each convolution and kernel_size is the size of the feature map which slides over the image to capture convolutions with a particular stride.
* Lines 33 and 34 are the fully connected layer with only one hidden layer where in_features is the number of neurons in first layer and out_features is the number of neurons at output.
* In line 36, count_neurons is initialized with the line 37 converts the image of the Doom game of dimensions 1x80x80 into a torch variable.
* From lines 38 to 40, max pooling of the convolutional layers is done and all the layers are activated using the [relu function](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning). Line 41 returns number of neurons which is required for fully connected 1 layer.
* In line 43, forward function is defined where the signals are fowarded into the CNN. From lines 44 to 50, convolutions with max pooling is done, those are activated using the relu activation in both fully connected layers and returns the signal.

![body](https://user-images.githubusercontent.com/35863175/46059467-f37d9800-c17c-11e8-8055-31e570b9d21f.JPG)
* In line 54, we define a class for making the body of our AI which is a softmax body with a temperature parameter T for controlling our AI's actions (outputs) in a particular state in the Doom environment and then in lines 59 to 62, this softmax is forwarded into the output, returning the actions of our AI.

![class ai](https://user-images.githubusercontent.com/35863175/46059470-f8dae280-c17c-11e8-86a1-3b89cc634858.JPG)
* In line 66, our AI class is created with a body and brain intitialized in the init function from lines 67 to 69.
* From lines 71 to 75, the call functions is created where, inputs are taken in the form of torch variable and this input is provided to the brain (AI algorithm), then the output of this brain is given to our body which defines the actions to be taken by the agent and is returned by the function.

![getting doom building ai and setting experience replay](https://user-images.githubusercontent.com/35863175/46059476-02644a80-c17d-11e8-9aac-4d0806ac55e3.JPG)
* From lines 81 to 83, we set our Doom environemnt using the image_processing module and the number of actions to be played by the agent in the Doom environment.
* From lines 86 to 88, we get our number of actions from the CNN and then this is provided to our softmax body which is going to implement these actions. The AI takes the brain and the actions to play in the Doom.
* In lines 91 and 92, the experience replay is initialized which is adaptive with the eligibility trace where instead of learning the Q-values every transition, it learns every 10 transitions. So learning is done in 10 steps and cummulative reward is calculated for the AI. This enables faster training of the model and this is called the N step eligibility trace. Capacity is the memory of our AI for storing the transitions.

![asynchronous n step q learning](https://user-images.githubusercontent.com/35863175/46075707-ea0a2500-c1a8-11e8-8e8f-d7871410aa7e.JPG)
* This is the algorithm of eligibility trace which is being implemented in this function.

![eligibility trace](https://user-images.githubusercontent.com/35863175/46059486-0bedb280-c17d-11e8-9e87-a72e03522646.JPG)
* Here the N step eligibility trace algorithm is implemented which is given in the research paper - Asynchronous Methods for Deep Reinforcement Learning in page 13 and is given in the study materials folder. Theta is the target and maximum of Q values will be calculated for the current state and action as well. We won't call this asynchronous (multiple agents) as we will be working with only one agent. Each transition of the series is having the structure - state, action, reward and done.
* In the line 95, the function eligibility_trace is initialized where the input to the function is a batch which consists of inputs and targets. AI will be trained in batches.
* Line 96 - gamma parameter which is a decay parameter needed in this algorithm.
* Lines 97 and 98 - inputs and targets are initialized as empty lists which will be filled later on.
* Line 99 - We start a loop for our series (10 transitions) in our batch.
* Line 100 - To get our cummulative reward, we need the state of the first and last transitions of our series which is derived here as a torch variable. Series0.state is first one and series-1.state is the last one.
* Line 101 - Output of the AI, i.e. predictions made by the AI.
* Line 102 - Cummulative reward is calculated according to the algorithm given in the page 13 of the research paper where if we reach the last state, reward is 0 (series-1.done) where done attribute (Boolean type) comes from [here](https://gym.openai.com/docs/), otherwise, we get the maximum of our Q values.
* Line 103 - We define a for loop in which we go from element before the last element (second last) to the first element (:-1 signifies that).
* Line 104 - Cummulative reward is updated according to the formula given in the research paper.
* Line 105 - Here we get the state of the first transition of our series.
* Line 106 - Q-value of the input state of the first transition.
* Line 107 - Here we are only interested in the Q -value for the action which was selected in the first step of the series and update target according to that only.
* Lines 108 and 109 - We only update the first step of the series because, we train the AI on 10 steps so we do not need the following steps after first step. Due to this reason, we append the first step only to our inputs and targets.
* Line 110 - Return the inputs and targets as torch tensors.

![moving average on 100 steps](https://user-images.githubusercontent.com/35863175/46059490-10b26680-c17d-11e8-8e2b-87c46024db3d.JPG)
* Line 113 - A class for moving average on 100 steps is defined and from lines 114 to 116, we create a function for class object as well as size which is the list of the rewards on which we are going to compute the average.
* Line 118 - A function to add cummulative rewards to the list of rewards.
* Line 119 and 120 - We are checking if the isinstance rewards is a list (means rewards are in the form of lists), then we add the list of rewards to rewards because both of them are lists and can be added in python.
* Line 121 and 122 - Else, when the rewards is not a list, then we append that single reward element to pur list of rewards.
* Lines 123 and 124 - Here if list of rewards gets bigger than 100 elements, then we delete the first of element of this list of rewards to make sure the list remains at 100 elements only.
* Lines 126 and 127 - Computing the moving average of the list of rewards of 100 steps at a time.

![training the ai](https://user-images.githubusercontent.com/35863175/46059497-190aa180-c17d-11e8-986c-941458cf47c2.JPG)
* Line 132 - We define our loss function which is the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error).
* Line 133 - We set the Adam optimizer like the self driving car case. Learning rate (lr) is taken to be small so that the algorithm do not converges fast by not exploring many options.
* Line 134 - Number of epochs for training. 1 epoch means going forward propagation into the NN and backward propagation into the NN.
* Line 135 - We start the loop from 1 to epoch +1 as the upper bound in python is excluded always.
* Line 136 - In this we have 200 steps running at each epoch.
* Line 137 - We sample some batches of 128 steps which are last run and stored into the memory and the learning will happen in these batches and inside these batches, the eligibility trace wiill be running for every 10 steps.
* Lines 138 and 139 - In this the we retrieve the inputs and targets as torch variables using the eligbility trace function applied into the batches.
* Line 140 - We get our predictions from the CNN as the loss is calculated between predictions and targets.
* Line 141 - We calculate the loss error between predictions and the targets.
* Line 142 - We initilaize our Adam optimizer here.
* Line 143 - backpropagation of loss error into the NN.
* Line 144 - Here the weights of our NN is updated.
* Lines 145 to 147 - We get he new cummulative rewards of the steps, add these new cummulative rewards to our moving average class and finally calculate the average reward.
* Line 148 - Prints the epoch and average rewards.
* Lines 149 to 151 - If the average reward is greater than 1500, our AI wins the Doom game!
* Line 154 - To close the Doom environement when the game is finished.


### Elucudation of experience_replay.py file
Here the experience replay which was implemented in the self driving car is implemented but now it is adaptive to the N step eleigibility trace. In this there are two classes, in the first one which is NStepProgress and is implementing the progress of the memory every N steps. Then we have the ReplayMemory class which makes the N step experience replay works and takes in the memory account of N steps for every transitions and not at each transition.

### Elucidation of image_preprocessing.py file
This python contains a class called PreprocessImage which is performing some image manipulations and mathematical functions onto the Doom image for its processing. There are two functions which are implementing the tasks such as converting image into black and white to reduce the memory overload and training time, resizing/cropping the image and doing some type conversions based on float variables.

## Running our Doom Gameplay
* Execute the code of ai.py file and it's done.
* You can see the outputs in the videos folder provided with this project folder.

## Acknowledgements
* Udemy online platform for sustaining this beautiful course on AI.
* Huge ton of thanks to Hadelin De Ponteves and Kirill Ermenko for creating this wonderful course on AI and Jordan Sauchuk (teaching assistant) for helping in the course throughout and clearing all the doubts.
* Lots of thanks to Zenodia Charpy for providing the appropriate instructions of OpenAI Gym on Windows, and those instructions also worked for me in my Ubuntu 16.04.
* Style of explanation of the code is inspired from Adrian Rosebrock. His [Linkedin](https://www.linkedin.com/in/adrian-rosebrock-59b8732a) and [website](https://www.pyimagesearch.com/author/adrian/)
* Lots of Thanks to Adit Deshpande for an excellent tutorial on CNNs.
* Google Images.












