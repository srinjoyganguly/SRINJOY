# Extended Kalman Filters

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

