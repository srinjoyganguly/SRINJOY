# Robotic Arm: Pick & Place

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

![KUKA_KR_210_cutout](https://user-images.githubusercontent.com/35863175/57874562-7b103800-782e-11e9-8d4c-6014e2b13c3b.png)

This is the image of a KUKA KR210 industrial robotic arm which is widely used in the industry for picking and placing objects. In this project we are going to simulate this pick and place operation using this KUKA arm in Robotic Operating System (ROS).

## Installation Instructions

Make sure you are using Robo-nd Virtual Machine or have Ubuntu+ROS installed locally. The VM of Udacity can be found here for [download](https://s3-us-west-1.amazonaws.com/udacity-robotics/Virtual+Machines/Lubuntu_071917/RoboVM_V2.1.0.zip) 
The password for this Virtual Machine is - robo-nd

### One time Gazebo setup step:
Check the version of gazebo installed on your system using a terminal:
```sh
$ gazebo --version
```
To run projects from this repository you need version 7.7.0+
If your gazebo version is not 7.7.0+, perform the update as follows:
```sh
$ sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
$ wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install gazebo7
```

Once again check if the correct version was installed:
```sh
$ gazebo --version
```
### For the rest of this setup, catkin_ws is the name of active ROS Workspace, if your workspace name is different, change the commands accordingly

If you do not have an active ROS workspace, you can create one by:
```sh
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```

Now that you have a workspace, clone or download this repo into the **src** directory of your workspace:
```sh
$ cd ~/catkin_ws/src
$ git clone https://github.com/udacity/RoboND-Kinematics-Project.git
```

Now from a terminal window:

```sh
$ cd ~/catkin_ws
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
$ cd ~/catkin_ws/src/RoboND-Kinematics-Project/kuka_arm/scripts
$ sudo chmod +x target_spawn.py
$ sudo chmod +x IK_server.py
$ sudo chmod +x safe_spawner.sh
```
Build the project:
```sh
$ cd ~/catkin_ws
$ catkin_make
```

Add following to your .bashrc file
```
export GAZEBO_MODEL_PATH=~/catkin_ws/src/RoboND-Kinematics-Project/kuka_arm/models

source ~/catkin_ws/devel/setup.bash
```
Whenever you open a new terminal window, it is recommended to run the source command before running anything.

For demo mode make sure the **demo** flag is set to _"true"_ in `inverse_kinematics.launch` file under /RoboND-Kinematics-Project/kuka_arm/launch

In addition, you can also control the spawn location of the target object in the shelf. To do this, modify the **spawn_location** argument in `target_description.launch` file under /RoboND-Kinematics-Project/kuka_arm/launch. 0-9 are valid values for spawn_location with 0 being random mode.

You can launch the project by
```sh
$ cd ~/catkin_ws/src/RoboND-Kinematics-Project/kuka_arm/scripts
$ ./safe_spawner.sh
```

If you are running in demo mode, this is all you need. To run your own Inverse Kinematics code change the **demo** flag described above to _"false"_ and run your code (once the project has successfully loaded) by:
```sh
$ cd ~/catkin_ws/src/RoboND-Kinematics-Project/kuka_arm/scripts
$ rosrun kuka_arm IK_server.py
```
Once Gazebo and rviz are up and running, make sure you see following in the gazebo world:

	- Robot
	
	- Shelf
	
	- Blue cylindrical target in one of the shelves
	
	- Dropbox right next to the robot
	

If any of these items are missing, report as an issue.

Once all these items are confirmed, open rviz window, hit Next button.

To view the complete demo keep hitting Next after previous action is completed successfully. 

Since debugging is enabled, you should be able to see diagnostic output on various terminals that have popped up.

The demo ends when the robot arm reaches at the top of the drop location. 

There is no loopback implemented yet, so you need to close all the terminal windows in order to restart.

In case the demo fails, close all three terminal windows and rerun the script.

[Rubric Points](https://review.udacity.com/#!/rubrics/972/view) are described individually below in detail.

### Kinematic Analysis

#### 1. Run the forward_kinematics demo and evaluate the kr210.urdf.xacro file to perform kinematic analysis of Kuka KR210 robot and derive its DH parameters.

I ran the forward_kinematics demo successfully and I evaluated the kr210.urdf.xacro file which actually contains two types of DH parameters called a(link length) and d(link offset). This will be helping  us to obtain our DH parameter table. 

![l01-19-l-forward-kinematics-01](https://user-images.githubusercontent.com/35863175/57877570-cb3ec880-7835-11e9-84ed-556002c10f1b.png)
The goal of forward kinematics (FK) is to calculate the pose of the end effector (EE) given all the joint angles, six in our case. The FK problem actually boild down to the composition of homogeneous transforms. We start with the base link and move link by link to the end effector.

Now I will go through the steps of calculating the DH parameters step by step. 

**Step 1 Label joints from 1 to n** 
<img width="859" alt="Step 1" src="https://user-images.githubusercontent.com/35863175/57875964-d1cb4100-7831-11e9-992b-0fe2d5cb96da.png">

**Step 2 Define joint axes**
<img width="868" alt="Step 2" src="https://user-images.githubusercontent.com/35863175/57876089-1ce55400-7832-11e9-8b02-6aeadceef07f.png">

**Step 3 Label links from 0 to n**
<img width="869" alt="Step 3" src="https://user-images.githubusercontent.com/35863175/57876155-469e7b00-7832-11e9-9354-2acc8482909f.png">

**Step 4 Define common normals and reference frame origins**
<img width="870" alt="Step 4" src="https://user-images.githubusercontent.com/35863175/57876237-89f8e980-7832-11e9-937a-6eb78bfb613a.png">

**Step 5 Add Gripper Frame** -  it's an extra frame, represents the point on the end effector that we actually care about. It differs from frame 6 only by a translation in z6 direction. 
<img width="875" alt="Step 5" src="https://user-images.githubusercontent.com/35863175/57876371-c9273a80-7832-11e9-9840-1d4790f0c479.png">

**Step 6 Definition of DH Parameters**
![DH Parameters Definition](https://user-images.githubusercontent.com/35863175/57876503-16a3a780-7833-11e9-877a-92ed381ff1c1.png)

* α (twist angle): the angle between Zi-1 and Zi measured about Xi-1 in a right hand sense.
* a (link length): the distance from Zi-1 to Zi measred along Xi-1 where Xi-1 is perpendicular to both.
* d (link offset): the signed distance from Xi-1 to Xi measured along Zi.
* θ (joint angle): the angle between Xi-1 and Xi measured about Zi in a right hand sense.

**Step 7 Location of each non-zero link lengths a and the link offsets d**
<img width="869" alt="Step 6" src="https://user-images.githubusercontent.com/35863175/57876637-769a4e00-7833-11e9-9620-0be1167397d8.png">

**Step 8 Obtaining values of a and d from kr210.urdf.xacro**
![URDF xacro file](https://user-images.githubusercontent.com/35863175/57876786-d4c73100-7833-11e9-86d8-2d8d1a89a41f.png)

**Step 9 Obtaining twist angles alpha by observing Zi-1 and Zi pair**

Z(i), Z(i+1) | alpha(i) |
--- | --- |
Z(0) ll  Z(1) | 0 | 
Z(1) ⟂ Z(2) | - pi/2 | 
Z(2) ll Z(3) | 0 |
Z(3) ⟂ Z(4) | -pi/2 |
Z(4) ⟂ Z(5) | pi/2 |
Z(5) ⟂ Z(6) | -pi/2 | 
Z(6) ll Z(G) | 0 | 

So, we get the DH parameter table as follows

Links | alpha(i-1) | a(i-1) | d(i-1) | theta(i)
--- | --- | --- | --- | ---
0->1 | 0 | 0 | L1 | qi
1->2 | - pi/2 | L2 | 0 | -pi/2 + q2
2->3 | 0 | 0 | 0 | 0
3->4 |  0 | 0 | 0 | 0
4->5 | 0 | 0 | 0 | 0
5->6 | 0 | 0 | 0 | 0
6->EE | 0 | 0 | 0 | 0
#### 2. Using the DH parameter table you derived earlier, create individual transformation matrices about each joint. In addition, also generate a generalized homogeneous transform between base_link and gripper_link using only end-effector(gripper) pose.

Links | alpha(i-1) | a(i-1) | d(i-1) | theta(i)
--- | --- | --- | --- | ---
0->1 | 0 | 0 | 0.75 | q1
1->2 | - pi/2 | 0.35 | 0 | q2 -pi/2
2->3 | 0 | 1.25 | 0 | q3
3->4 |  -pi/2 | -0.054 | 1.50 | q4
4->5 | pi/2 | 0 | 0 | q5
5->6 | -pi/2 | 0 | 0 | q6
6->EE | 0 | 0 | 0.303 | q7

The code sample for this DH parameter is given below
![DH Table Code](https://user-images.githubusercontent.com/35863175/57877503-a21e3800-7835-11e9-88e7-5ddf27cb975c.png)


#### 3. Decouple Inverse Kinematics problem into Inverse Position Kinematics and inverse Orientation Kinematics; doing so derive the equations to calculate all individual joint angles.

And here's where you can draw out and show your math for the derivation of your theta angles. 

![alt text][image2]

### Project Implementation

#### 1. Fill in the `IK_server.py` file with properly commented python code for calculating Inverse Kinematics based on previously performed Kinematic Analysis. Your code must guide the robot to successfully complete 8/10 pick and place cycles. Briefly discuss the code you implemented and your results. 


Here I'll talk about the code, what techniques I used, what worked and why, where the implementation might fail and how I might improve it if I were going to pursue this project further.  


And just for fun, another example image:
![alt text][image3]

