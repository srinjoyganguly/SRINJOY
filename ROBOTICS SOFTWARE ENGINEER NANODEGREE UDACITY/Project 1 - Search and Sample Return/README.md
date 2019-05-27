[//]: # (Image References)
[image_0]: ./misc/rover_image.jpg
[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
# Search and Sample Return Project


![rover_image](https://user-images.githubusercontent.com/35863175/58412820-e65ed300-8094-11e9-9c1f-a5a2a8102f63.jpg)


This project is modeled after the [NASA sample return challenge](https://www.nasa.gov/directorates/spacetech/centennial_challenges/sample_return_robot/index.html) and it will give you first hand experience with the three essential elements of robotics, which are perception, decision making and actuation.  You will carry out this project in a simulator environment built with the Unity game engine.  

## Installation Instructions and Project Details

**The `code` folder contains all the important files - `Rover_Lab_Notebook.ipynb`, `supporting_functions.py`, `decision.py`, `perception.py` and `drive_rover.py`

## The Simulator
The first step is to download the simulator build that's appropriate for your operating system.  Here are the links for [Linux](https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Linux_Roversim.zip), [Mac](	https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Mac_Roversim.zip), or [Windows](https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Windows_Roversim.zip).  

You can test out the simulator by opening it up and choosing "Training Mode".  Use the mouse or keyboard to navigate around the environment and see how it looks.

## Dependencies
You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/ryan-keenan/RoboND-Python-Starterkit). 


Here is a great link for learning more about [Anaconda and Jupyter Notebooks](https://classroom.udacity.com/courses/ud1111)

## Recording Data
I've saved some test data for you in the folder called `test_dataset`.  In that folder you'll find a csv file with the output data for steering, throttle position etc. and the pathnames to the images recorded in each run.  I've also saved a few images in the folder called `calibration_images` to do some of the initial calibration steps with.  

The first step of this project is to record data on your own.  To do this, you should first create a new folder to store the image data in.  Then launch the simulator and choose "Training Mode" then hit "r".  Navigate to the directory you want to store data in, select it, and then drive around collecting data.  Hit "r" again to stop data collection.

## Data Analysis
Included in the IPython notebook called `Rover_Lab_Notebook.ipynb` are the functions from the lesson for performing the various steps of this project.  The notebook should function as is without need for modification at this point.  To see what's in the notebook and execute the code there, start the jupyter notebook server at the command line like this:

```sh
jupyter notebook
```

This command will bring up a browser window in the current directory where you can navigate to wherever `Rover_Lab_Notebook.ipynb` in the `code` folder is and select it.  Run the cells in the notebook from top to bottom to see the various data analysis steps.  

The last two cells in the notebook are for running the analysis on a folder of test images to create a map of the simulator environment and write the output to a video.  These cells should run as-is and save a video called `test_mapping.mp4` to the `output` folder.  This should give you an idea of how to go about modifying the `process_image()` function to perform mapping on your data.  

## Navigating Autonomously
The file called `drive_rover.py` is what you will use to navigate the environment in autonomous mode.  This script calls functions from within `perception.py` and `decision.py`.  The functions defined in the IPython notebook are all included in`perception.py` and it's your job to fill in the function called `perception_step()` with the appropriate processing steps and update the rover map. `decision.py` includes another function called `decision_step()`, which includes an example of a conditional statement you could use to navigate autonomously.  Here you should implement other conditionals to make driving decisions based on the rover's state and the results of the `perception_step()` analysis.

`drive_rover.py` should work as is if you have all the required Python packages installed. Call it at the command line like this: 

```sh
python drive_rover.py
```  

Then launch the simulator and choose "Autonomous Mode".  The rover should drive itself now!  It doesn't drive that well yet, but it's your job to make it better!  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results!  Make a note of your simulator settings in your writeup when you submit the project.**

**Simulator Settings :**
* Screen Resolution : 1280x600
* Graphics Quality : Fastest
* Select Monitor : Display 1
* Windowed is tick marked

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points are described in detail

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.   


### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

**The Jupyter Notebook named - Rover_Lab_Notebook.ipynb can be found inside the folder `code`**

First of all, we a take a quick look at the image data which is provided already by the Udacity team. **I have used the data already given by the Udacity team in their workspace.**

The image is given as follows :


![Quick look at the data](https://user-images.githubusercontent.com/35863175/58413639-f24b9480-8096-11e9-90f2-06a08899c8f8.png)

After this, we read in an image and do color thresholding, to identify a yellow rock sample, to detect navigable path and the obstacles also as shown as follows :


![Calibration Data](https://user-images.githubusercontent.com/35863175/58413866-7867db00-8097-11e9-8b9c-05972a6de5e9.PNG)

In the code, I have done as follows :

![color thresholding code](https://user-images.githubusercontent.com/35863175/58414079-f1ffc900-8097-11e9-8312-ce98e9495fb1.PNG)

Now, I perform the Perspective Tranformation which just tells us how the image looks from the upper view. To remove some black spots which might be confused with obstacles, I decided to put a field of view parameter as well in this perspective transformation. Here is the image as follows :

![Perspective Transformation](https://user-images.githubusercontent.com/35863175/58414188-4d31bb80-8098-11e9-850a-061339d0b399.PNG)

And, in the code I have done it as follows :

![perspective transform code](https://user-images.githubusercontent.com/35863175/58414222-65a1d600-8098-11e9-9144-ed88dc236a37.PNG)

After this step, I performed color thresholding of the perspective transformed image, as follows :

![Color Thresholding](https://user-images.githubusercontent.com/35863175/58414286-8cf8a300-8098-11e9-85a5-f19000104a2d.PNG)


Then I performed some coordinate transformations such as converting picel positions into rover coordinates, converting pixel coordinates to polar corrdinates, applying rotation to the pixels and translation as well, and finally transforming the pixels to the world coordinates. After doing all the transformations, I got as follows :

![Coordinate Transformations](https://user-images.githubusercontent.com/35863175/58414421-f8427500-8098-11e9-9c3b-d929db099771.PNG)


#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

The code for the `process_image()` function is given as follows and is properly commented so as to ease understanding :

![process image code 1](https://user-images.githubusercontent.com/35863175/58414529-40619780-8099-11e9-97cb-b44f23482b5f.PNG)


![process image code 2](https://user-images.githubusercontent.com/35863175/58414541-4a839600-8099-11e9-8194-3d1b62113e4d.PNG)


![process image code 3](https://user-images.githubusercontent.com/35863175/58414554-55d6c180-8099-11e9-94eb-0b38a19b9c65.PNG)

After filling out the entire `process_image()` function, I used the images provided by the Udacity in their workspace to create a test_mapping.mp4 video which can be found inside the folder `output`. I ran the `process_image()` function on the images using the `moviepy` to create the video.

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

##### The perception.py file modifications

Here the code for color thresholding and perspective transformation was used form the Jupyter notebook which has been explained above. 

Here is the code for the perception step :

![perception_code_1](https://user-images.githubusercontent.com/35863175/58415414-05ad2e80-809c-11e9-84b0-edaad1dd6a95.PNG)


![perception_code_2](https://user-images.githubusercontent.com/35863175/58415434-12ca1d80-809c-11e9-907b-512bb96a8061.PNG)

I also applied the to_polar_coords() function to the sample rock 'x' and 'y' pixels to provide the rover distance and direction to where the rock samples are, for steering guidance.

##### The decision.py file modifications

I have made these following changes to the `decision_step()` function to provide the extra capability to locate and steer towards rock samples when found, stop when near a sample, and pickup sample when it has stopped in front of the rock sample. The code is given as follows :

![decision_code_1](https://user-images.githubusercontent.com/35863175/58415536-63da1180-809c-11e9-8560-5635a6cc69bd.PNG)

![decision_code_2](https://user-images.githubusercontent.com/35863175/58415550-6dfc1000-809c-11e9-9046-02abc99a6d24.PNG)

![decision_code_3](https://user-images.githubusercontent.com/35863175/58415567-781e0e80-809c-11e9-8732-bcacdcd4c187.PNG)

![decision_code_4](https://user-images.githubusercontent.com/35863175/58415590-88ce8480-809c-11e9-9fcd-f2d386aaedb0.PNG)

The changes done above are mostly based on conditional statements to add more functionality to the rover such as stopping slowly when rock samples are located, when encountered a non naviagble path, then it shoud steer until it gets navigable terrain.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

Here is the link of a YouTube video regarding my rover detecting rock samples successfully and mapping the environment more than 40% while maintaining its fidelity between 60% and 70% successfully.

https://youtu.be/d1iuw4E4Ao0

You might observe that sometimes I manually steer the rover. That is because it sometimes gets stuck and improvements can be made in this in the future related to it's decision making capabilities. **Further improvements have been mentioned below.**

### Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

#### My Approach

**The `decision.py` file was completed using if and else and elif statements for making the decisions for the throttle, braking and steering. This is  hard coding style where I implemented a simple decision tree based on what the robot perceives consisting of conditional statements.** 

**The `perception.py` file was completed with th help of my `process)image()` function which I implemented in the Jupyter Notebook. In that I applied color thresholding, then perspective transformation, then coordinate transformation on the pixels of samples and obstacles. Finally, I used them to update the locations of the rock samples so that it can be detected and when detected it can be picked up also**

#### Improvements

* **Sometimes the rover is running around in circles, so to prevent this we can do some improvements such as making the rover return to a home position.**

* **The rover occasionally, get on top of the rocks after detecting them, so the decision.py can be changed so that, the rover stops at a certain minimum distance from the rock sample to pick it up.**
