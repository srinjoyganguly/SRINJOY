# SELF DRIVING CAR SIMULATION - DEEP REINFORCEMENT LEARNING (Deep Q-Learning)

![self_driving_car](https://user-images.githubusercontent.com/35863175/45965926-700e5a80-c047-11e8-97aa-614116c83092.jpg)

In this project we are going to simulate a self driving car in a Kivy environment. We will be training the brain of the car using Deep Q learning AI algorithm and will be drawing the map on the Kivy map environment for our car to navigate in that environment. We will be testing various maps for checking the performance of our car. Our car is having three sensors which will be detecting sand (obstacle) in front, left and right directions. These sensors will be generating signals which will be densities of the sand and will be used for the processing of state, action, and reward for the car in a particular state. Our car is the agent which will navigate and explore the Kivy environment and undertake various actions based on it's reward and state. We penalize the car whenever it goes towards the dge of the map or touches any sand and in this way the car trains itself and learns the path and the actions to take.

## Prerequisites for this project

To understand this advanced project, there are some prerequisites which needs to be met so that you can enjoy the benefits of this project. Let's summarize them - 
* Understanding of [Artificial Neural Networks](https://www.tutorialspoint.com/artificial_neural_network/) and [Deep Learning](https://www.deeplearningbook.org/). Many online courses can be found on these topics from Coursera, Udemy, edX and Udacity.
* The online course of AI from UC Berkeley, which you can find [here](http://ai.berkeley.edu/lecture_videos.html) - most important topics are - reinforcement learning, Bellman equation, Markov Decision Process, Q-Learning, Deep Q-Learning, Temporal Difference and experience replay.
* Programming experience in Python and Pytorch library which will be used in this heavily. Pytorch dcumentation can be found [here](https://pytorch.org/docs/stable/index.html).

## Installation Instructions
### For Windows
1. Install Anaconda software - Python 3.6 version from [here](https://www.anaconda.com/download/). This software comes with the Python IDE and lots of more libraries for data science and machine learning such as Numpy, pandas, matplotlib, scipy etc.
2. Open the Anaconda Prompt, which acts like a terminal in windows environment similar to an Ubuntu or Mac system.
* It's always better to create a separate environement for your project (so that the project contents/settings do not interfere with your original root folder) by running this command - conda create -n env-name python=3.x anaconda. In this x is for the version for example 2.7, 3.5 or 3.6 etc and activate the environment with the commands shown in the terminal after you make the environment.
3. Install Pytorch library using the instructions given [here](https://pytorch.org/get-started/locally/). It is recommended to install the No CUDA - CPU version of Pytorch if you do not have the proper Nvidia graphics card drivers installed or do not have any graphics card. If you have the proper graphics card and it's driver then you are welcome to install the CUDA versions of Pytorch according to your version. To install previous versions of Pytorch click [here](https://pytorch.org/get-started/previous-versions/) and install using pip instructions. Pip ensures that you install everything in your environment only and do not interferes with your root system, however if you install using conda, then it is installed for all the enviornemnts and this interferes with all other environments in your system.
4. Download https://files.pythonhosted.org/packages/12/c1/058b2046fe0c83f489600a134720779c9e53123629f46d0239d345743042/Kivy-1.10.1-cp36-cp36m-win_amd64.whl
5. Go to the download folder in your Anaconda Prompt console ("cd folder name"), the folder where you downloaded the file from step 4.
6. Enter the following commands in the Anaconda Prompt console to install Kivy:-
* pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew
* pip install kivy.deps.gstreamer
* pip install Kivy-1.10.1.dev0-cp36-cp36m-win_amd64.whl
7. In Spyder go to Tools -> Preferences -> Run
8. Set Console -> Execute in an external system terminal
9. In case of any error, please consult the debugging files provided with the project. A common error is - TypeError: multinomial() missing 1 required positional arguments: "num_samples". In this case downgrading the pytorch version will help. The verison must be less than 0.4.0

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
* [Numpy](http://www.numpy.org/) is imported in line 5 and is used for the scientific computation and linear algebra. 
* In line 6, [random library](https://www.pythonforbeginners.com/random/how-to-use-the-random-module-in-python) is imported which is used for picking random elements from a list or from different batches, [os](https://www.pythonforbeginners.com/os/pythons-os-module) in line 7 is for using the operating system based functions and for interfacing python with OS such as Windows, Mac or  linux. 
* From lines 8 to 11, we import all [torch modules](https://pytorch.org/docs/stable/nn.html) for creating our neural network, calculating loss functions and optimizers. 
* Lines 12 and 13 is for conversion of torch tensors into variables which contain gradients.

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
* Replay memory is the implementation of experience replay in which we put some amount of previous (past) states in the memory of the agent navigating the environment with the current state of the agent. After this, we take random batches of these samples of past states to make the next update of the state/action of the agent. In line 33, class is made which inherits the [object class from python to make the code work for both 2.x and 3.x python versions](https://stackoverflow.com/questions/4015417/python-class-inherits-object). 
* From lines 35 to 37, init function is created with capacity variable (let it be 100000) which stores the maximum transitions in the memory of events and memory variable for the list of past events.
* From lines 39 to 42, push function is created to append the events to the memory of the agent and delete the oldest transition of events for maintaining the capacity of the memory fixed. Event is tuple of 4 elements - last state, new state, last action and last reward.
* From lines 44 to 46, the batches are sampled according to state, action and reward using zip and then returned as torch variables. This is done to make everything aligned i.e. in each row - state, action and reward correspond to same time t and eventually we get list of all batches properly aligned in this fashion. From these samples the model will learn.

![dqn_init_select_action](https://user-images.githubusercontent.com/35863175/45941667-edf94400-bffc-11e8-9346-35f78a469fb6.JPG)
* Deep Q Network (Dqn) class is created which has various functions to implement the Deep Q-Learning AI algorithm and will form the brain of our self driving car. Since this class is quite large with lots of functions inside it, we have divided it into chunks for better explanation and understanding.
* In line 50,  we start our Dqn class and in line 52 we initializ ethe init function taking arguments self, input_size, nb_action and gamma- delay coefficient(this comes from the equation the Deep Q Learning model).
* In line 54, an evolving mean of last 100 rewards to evaluate the evolution of performance of the AI.
* Line 55 - model of NN for our Deep Q Learning.
* Line 56 - capacity of 100000 is initialized in the ReplayMemory.
* Line 57 - connects optimizer to the Deep Q Network and we use the [Adam optimizer](https://medium.com/@nishantnikhil/adam-optimizer-notes-ddac4fd7218).
* Line 58 - Last state is created which is a vector of 5 dimensions - three signals of the three sensors - straight, left and right and orientation and minus orientation. A fake dimension corresponding to batch is also included using the unsqueeze(0) method as for making this a torch tensor as NN accepts inputs in batches only. 0 corresponds to fake dimesnions of the state.
* Lines 59 and 60 creates the variables of the last action and reward.
* In line 62, select_action function is made having inputs as self and state (input state). So, in 63, we generate a distribution of probablities for each Q values corresponding to the three actions - straight, left and right using the softmax function which will attribute large probability to highest Q value. We have three actions to play for each input state. This also enables to keep exploring other actions as well. Volatile is True, which suggests that gradients will not be considered as input to the NN since NN do not require gradients at it's input. Here 30 is the temperature parameter which decides probability of winning Q value. So higher this value is better our self driving car becomes and this tells about the certainity about a particular action agent needs to play. 
* Lines 64 and 65 has the actions generated and returned for the agent to play - straight, left or right for a particular input state. Multinomial returns pytorch variable with fake batch.

![learn and update](https://user-images.githubusercontent.com/35863175/45941675-f5b8e880-bffc-11e8-9b18-84b4b9570c0c.JPG)
* Learning (training) and updation of weights of the self driving car network. From lines 67 to 69, we start implementing the learn function which takes input parameters in batches since NN works with batchs as input only, get outputs (simple vector, not tensor) for the batch_states which are inputs of the NN and next_outputs which gives us the maximum Q value of the next state represented by all actions. In the gather function, 1 is chosen with batch_action because we only want the action which was chosen to play and not all actions and this gathers the best action each time for each of the input states. Unsqueeze(1) corresponds to fake dimensions of action which is killed using squeeze(1) as we want outputs as vectors and not in batches (tensors). Detach is used to detach all outputs of the model as we have several states in this batch_next_state which has batches of all next states of all transitions from our random sample memory. Max(1) says that to maximize Q value for action in the next state, so we write 0 for state.
* Line 70 - formula of target is implemented.
* Line 71 - Temporal Difference loss is calculated in this step. [Smooth_l1_loss](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwjLz_bPwdPdAhXMso8KHQL9DWQQFjAAegQICBAC&url=https%3A%2F%2Fgithub.com%2Frbgirshick%2Fpy-faster-rcnn%2Ffiles%2F764206%2FSmoothL1Loss.1.pdf&usg=AOvVaw0vqtGZqZTn8jnnVXjrPkue) is the loss function for Deep Q Learning.
* Line 72 - Reinitializes optimizer at each iteration of the loop.
* Line 73 - Loss function calculated in line 71 is back propagated into the network and retain_variables = True for freeing memory which will improve the training performance.
* Line 74 - updates the weights of the NN using the optimizer.
* In Line 76, the update function is created with self, reward and the new_signal is used for the updating the AI action, the last action becomes the new action, the last state becomes the new state and last reward becomes the new reward and finally we append these new transitions to memory and update the reward window as well. This function will be connecting the map.py file with ai.py file.
* Line 77 - This new_state depends upon the new_signal (state) which is detected by the sensors when the agent reaches a particular new state.
* Line 78 - In this we append the new transition just made in the previous line 77 into our memory using the push function made before and this now conatins the last state, new state, last action played and the last reward.
* Line 79 - Now, since we came to a new state after playing an action, so naturally, we need to play an action and this only is done in this step by using the select_action function with the new_state we have just reached.
* From lines 80 to 82, we are going to make our AI learn on the basis of its last 100 events picked randomly from the memory of 100000 events, so, we check that if the memory is having events more than 100 or not and if that's true, we collect the samples of the events in batches and start learning on the batches.
* From lines 83 to 86, all the last action , state, reward and the reward window are updated to new ones.
* From lines 87 to 89, we just make our size of reward window fixed, so that it doesn't changes and finally return our action.

![score save and load](https://user-images.githubusercontent.com/35863175/45941706-05383180-bffd-11e8-9db8-e1826d1152da.JPG)
* Scores will be assigned to the car according to it's performance. If it hits the sand drawnby us, negative rewards will be provided and when it goes smoothly without any hit, it will generate more positive rewards. Lines 91 and 92 are the function which returns the score of this agent.
* Lines 94 to 97 - This function is created to save the state of the brain of our self driving car, so that we can again run it from it's previous saved state when we load it using the load function.
* Lines 99 to 107 is the load function. Here first condition is checked if the last state of our brain exists or not and if it does, then, the checkpoint, model and optimizer of our self driving car is loaded.
* If we don't want the car to have a brain, we need to set the temperature parameter to 0. For better performance, this parameter should be high, but making it too high will make the AI to ignore exploring other actions, so this becomes a trade off point.

### Elucidation of map.py
The file called map_commented.py contains detailed explanations of each and every line of the code used in our project.

## Running our Self Driving Car - Outputs

For running the output, just open the map.py file and run it. To make the self driving car more better, some additional hidden layers can be added into the Network function in ai.py and in map.py the rewards can be adjusted from the lines 140 onwards. You can choose different negative rewards for the car if it touches the sand or it goes if it  touches the corners of the map. Also, try tweaking the temperature parameter.

![track1](https://user-images.githubusercontent.com/35863175/46009096-199f2b80-c0dc-11e8-966d-3055db938fec.gif)

![track2](https://user-images.githubusercontent.com/35863175/46009105-24f25700-c0dc-11e8-9fd7-12290ea9621d.gif)

![track3](https://user-images.githubusercontent.com/35863175/46009131-39365400-c0dc-11e8-90ce-d72eb1f71777.gif)


## Acknowledgements
* Udemy online platform for sustaining this beautiful course on AI.
* Huge ton of thanks to Hadelin De Ponteves and Kirill Ermenko for creating this wonderful course on AI
* Lots of thanks to Christian Rosan for providing the appropriate instructions of Kivy installation on Windows and Igor Brega and Sebastian Zarzycki for their contribution in creating challenging maps and making some valuable additions to the program.
* Style of explanation of the code is inspired from Adrian Rosebrock. His [Linkedin](https://www.linkedin.com/in/adrian-rosebrock-59b8732a) and [website](https://www.pyimagesearch.com/author/adrian/).
* Google Images.













