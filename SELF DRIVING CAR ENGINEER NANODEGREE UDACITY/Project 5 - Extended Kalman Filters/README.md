# Extended Kalman Filters

In this project, I implemented an Extended Kalman Filter in C++ language. The simulator provided by the Udacity generates noisy LIDAR and RADAR measurements of the position and velocity of an object and using my EKF implementation I performed a sensor fusion of the LIDAR and RADAR data to predict the positiona dn velocity of the object.

Here is the EKF map shown : 

![Kalman Filter Algorithm Map](https://user-images.githubusercontent.com/35863175/58876128-ccec0580-86ea-11e9-8b3f-ae238df66100.png)


This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Tips for setting up your environment can be found in the classroom lesson for this project.

Note that the programs that need to be written to accomplish the project are src/FusionEKF.cpp, src/FusionEKF.h, kalman_filter.cpp, kalman_filter.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---

## Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

## The [Project Rubric](https://review.udacity.com/#!/rubrics/748/view) Points are addressed below - 

## Compiling

### Your code should compile.

The code compiles successfully without any errors with cmake and make.


This is shown below through images :


![cmake_and_make_1](https://user-images.githubusercontent.com/35863175/58874913-b85a3e00-86e7-11e9-9e04-16283c5afb62.PNG)


![cmake_and_make_2](https://user-images.githubusercontent.com/35863175/58874928-c6a85a00-86e7-11e9-8d80-541d8a15012d.PNG)


![cmake_and_make_3](https://user-images.githubusercontent.com/35863175/58874951-d2941c00-86e7-11e9-943c-af9ea8acd295.PNG)

After executing ./ExtendedKF, it ran successfully. First I executed the command, then I opened the simulator and started running the EKF and in the terminal it showed connected as follows :


![Connected](https://user-images.githubusercontent.com/35863175/58875003-f9eae900-86e7-11e9-8360-9b63dd2480f4.PNG)

## Accuracy

### px, py, vx, vy output coordinates must have an RMSE <= [.11, .11, 0.52, 0.52] when using the file: "obj_pose-laser-radar-synthetic-input.txt" which is the same data file the simulator uses for Dataset 1.

My EKF algorithm ran quite successfully on the Dataset 1 and I got the following accuracy : 

RMSE <= [0.0973, 0.0855, 0.4513, 0.4399]

Here is the screenshot of my output : 


![EKF_Dataset1](https://user-images.githubusercontent.com/35863175/58875078-34ed1c80-86e8-11e9-9a45-33501f8b9b76.PNG)

## Follows the Correct Algorithm

### Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons.

The main Kalman Filter implementation can be found inside the folder src and is named as kalman_filter.cpp.

The predict function has been filled successfully and can be found from lines 25 to 32. The update function for LIDAR has been filled in from lines 34 to 40. The updateEKF function for RADAR is filled in from lines 42 to 65. 

To increase the code readability, I made a small function called UpdateWithY from lines 67 to 77 to calculate H_transpose, S_inverse, Kalman gain (K), new state and P as these are used in the update and updateEKF functions. This new function addition has been accounted for inside the kalman_filter.h file as well and I have defined it as private inside the class.

### Your Kalman Filter algorithm handles the first measurements appropriately.

The first measurement has been handled carefully in the file called FusionEKF.cpp from lines 59 to 103 and can be found inside the folder src.

### Your Kalman Filter algorithm first predicts then updates.

The calling of predict function can be found in the line 141in the file FusionEKF.cpp and the update mechanism has been implemented in the lines 153 to 165. So, it is concluded easily that my KF first predicts then it updates.

### Your Kalman Filter can handle radar and lidar measurements.

In the FusionEKF.cpp file – 

The measurement for LIDAR and RADAR is been implemented in the lines 59 to 103, where the first measurement has also been taken care of. Here conditional statements are used for detecting RADAR or LIDAR and then appropriate calculations have been implemented.

The update has been implemented in lines 153 to 165, both for RADAR and LIDAR. Here also conditional statements have been used to detect LIDAR or RADAR and then update equations have been calculated.

## Code Efficiency

### Your algorithm should avoid unnecessary calculations.

In the FusionEKF.cpp file – 

From lines 126 to 140, I have calculated the Q matrix, and I have used proper pre calculated values to compute the matrix. This is very good for code optimization as I did not defined any loop to calculate the matrix, didn’t calculated it repeatedly and avoided using any complex data structures as well. 

