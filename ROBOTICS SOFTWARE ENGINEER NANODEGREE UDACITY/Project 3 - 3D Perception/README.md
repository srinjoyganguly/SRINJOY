# 3D Perception

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

![PR2 Robot](https://user-images.githubusercontent.com/35863175/57986393-3997ac80-7a29-11e9-93de-6bc2d79df928.png)

The above image shows a PR2 robot which has been developed by Willow Garage. Similar to the KUKA KR210 robot, this also does pick and place operations but this robot is more advanced and complex than KUKA in the sense that this has perception system being build into this with the help of sensors such as camera, Lidar etc. 

In this project, I will be building a perception pipeline for a PR2 robot using the ROS to simulate the robotic perception system. I will do this by creating a perception system which can successfully detect objects in a cluttered table top from three different world's - World 1(Containing 3 objects), World 2(containing 5 objects) and World 3(containing 8 objects). The PR2 robot is fitted with a RGBD camera which essentially captures the RGB (Colour) as well as Depth information also to effciently obtain a picture of 3D World. This is very useful for Point Cloud Data processing as they work on 3D coordinates. 

## Installation Instructions
It is highly recommended to download the Udacity Virtual Machine which has all the packages and dependencies installed which are to be used in this project. You can download it [here](https://s3-us-west-1.amazonaws.com/udacity-robotics/Virtual+Machines/Lubuntu_071917/RoboVM_V2.1.0.zip)
The password for this Virtual Machine is - robo-nd

For this setup, catkin_ws is the name of active ROS Workspace, if your workspace name is different, change the commands accordingly
If you do not have an active ROS workspace, you can create one by:

```sh
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```

Now that you have a workspace, clone or download this repo into the src directory of your workspace:
```sh
$ cd ~/catkin_ws/src
$ git clone https://github.com/udacity/RoboND-Perception-Project.git
```
### Note: If you have the Kinematics Pick and Place project(Project 2 can be found [here](https://github.com/srinjoyganguly/SRINJOY/tree/master/ROBOTICS%20SOFTWARE%20ENGINEER%20NANODEGREE%20UDACITY/Project%202%20-%20Robotic%20Arm:%20Pick%20%26%20Place)) in the same ROS Workspace as this project, please remove the 'gazebo_grasp_plugin' directory from the `RoboND-Perception-Project/` directory otherwise ignore this note. 

Now install missing dependencies using rosdep install:
```sh
$ cd ~/catkin_ws
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
```
Build the project:
```sh
$ cd ~/catkin_ws
$ catkin_make
```
Add following to your .bashrc file
```
export GAZEBO_MODEL_PATH=~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/models:$GAZEBO_MODEL_PATH
```

If you haven’t already, following line can be added to your .bashrc to auto-source all new terminals
```
source ~/catkin_ws/devel/setup.bash
```

To run the demo:
```sh
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts
$ chmod u+x pr2_safe_spawner.sh
$ ./pr2_safe_spawner.sh
```
![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)



Once Gazebo is up and running, make sure you see following in the gazebo world:
- Robot

- Table arrangement

- Three target objects on the table

- Dropboxes on either sides of the robot


If any of these items are missing, please report as an issue on [the waffle board](https://waffle.io/udacity/robotics-nanodegree-issues).

In your RViz window, you should see the robot and a partial collision map displayed:

![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Proceed through the demo by pressing the ‘Next’ button on the RViz window when a prompt appears in your active terminal

The demo ends when the robot has successfully picked and placed all objects into respective dropboxes (though sometimes the robot gets excited and throws objects across the room!)

Close all active terminal windows using **ctrl+c** before restarting the demo.

You can launch the project scenario like this:
```sh
$ roslaunch pr2_robot pick_place_project.launch
```
[Rubric Points](https://review.udacity.com/#!/rubrics/1067/view) of the project are explained below in detail

All the code sections discussed here are from the file **Project_3D_Perception.py** file which can be found inside the folder **Python Code Files**. All the required Python files are also present in that folder.

## Filtering and Segmentation

### Point Cloud Filtering - Voxel Grid Downsampling, PassThrough Filtering, RANSAC Plane Fitting, Extract Indices, Outlier Removal Filter

#### Voxel Grid Downsampling Filter
Filtering is usually done to improve the quality of any data. Here we are using various filtering techniques to improve the quality of our Point Cloud data because all the sensors are inherently subjected to noise due to manufacturing defects, outside temperature etc.

RGB-D cameras provide feature rich and particularly dense point clouds, meaning, more points are packed in per unit volume than, for example, a Lidar point cloud. Running computation on a full resolution point cloud can be slow and may not yield any improvement on results obtained using a more sparsely sampled point cloud.

So, in many cases, it is advantageous to downsample the data. In particular, we are going to use a VoxelGrid Downsampling Filter to derive a point cloud that has fewer points but should still do a good job of representing the input point cloud as a whole.

A voxel grid filter allows you to downsample the data by taking a spatial average of the points in the cloud confined by each voxel. 

Below, I show the screenshots of the Voxel Grid Downsampling method
![Voxel Grid Downsampling](https://user-images.githubusercontent.com/35863175/57986658-934da600-7a2c-11e9-8d9d-86eb89e3309c.png)

![Voxel Grid Downsampling2](https://user-images.githubusercontent.com/35863175/57986676-dc9df580-7a2c-11e9-917e-82447aa19605.png)

And, here is the code peratining to the Voxel Grid Downsample filter
![Voxel Grid Downsampling Code](https://user-images.githubusercontent.com/35863175/57986699-27b80880-7a2d-11e9-9192-0486046f1746.png)

I have chosen LEAF SIZE of 0.005 after experiments and got good results. Other leaf sizes such as 0.001 or 0.002 etc will also work.

#### PassThrough Filter
The Pass Through Filter works much like a cropping tool, which allows you to crop any given 3D point cloud by specifying an axis with cut-off values along that axis. The region you allow to pass through, is often referred to as region of interest.

Here is the image of PassThrough Filtering where we only retain the tabletop and the objects and discard other information.


![Pass Through Filtering](https://user-images.githubusercontent.com/35863175/57986755-e2480b00-7a2d-11e9-9ce4-9b75cd0e6598.png)

The code for this is given as follows


![Pass Through Filter Object](https://user-images.githubusercontent.com/35863175/57986784-1b807b00-7a2e-11e9-80d0-dd11cd14e55f.png)

I have chosen axis_min of 0.6 and axis_max of 1.1 for z filtering and for y filtering I chose -0.456 to 0.456 after some experimentation.

#### RANSAC Plane Fitting
RANSAC is an algorithm, that you can use to identify points in your dataset that belong to a particular model. We will remove the table itself from the scene by using RANSAC!

The RANSAC algorithm assumes that all of the data in a dataset is composed of both inliers and outliers, where inliers can be defined by a particular model with a specific set of parameters, while outliers do not fit that model and hence can be discarded. 

The RANSAC algorithm mainly involves performing two iteratively repeated steps on a given data set: Hypothesis and Verification. First, a hypothetical shape of the desired model is generated by randomly selecting a minimal subset of n-points and estimating the corresponding shape-model parameters. A minimal subset contains the smallest number of points required to uniquely estimate a model.

Once a model is established, the remaining points in the point cloud are tested against the resulting candidate shape to determine how many of the points are well approximated by the model.

After a certain number of iterations, the shape that possesses the largest percentage of inliers is extracted and the algorithm continues to process the remaining data.

Here is the code for RANSAC

![RANSAC Plane Segmentation Code](https://user-images.githubusercontent.com/35863175/57986886-4b7c4e00-7a2f-11e9-84a8-d8b039af28f1.png)

I have chosen max_distance as 0.01 after some experimentation.

#### Extract Indices
the ExtractIndices Filter allows us to extract points from a point cloud by providing a list of indices. With the RANSAC fitting we just performed, the output inliers corresponds to the point cloud indices that were within max_distance of the best fit model.

So here is the image of the Extracted Inliers
![Extracted Inliers](https://user-images.githubusercontent.com/35863175/57986920-c5143c00-7a2f-11e9-82bc-f7b83c639c92.png)

And here is the image of Extracted Ouliers
![Extracted Outliers](https://user-images.githubusercontent.com/35863175/57986928-dcebc000-7a2f-11e9-9b86-814e37ca76e2.png)

#### Outlier Removal Filter
One of the filtering techniques used to remove outliers such as dust in the environment, humidity in the air, or presence of various light sources lead to sparse outliers is to perform a statistical analysis in the neighborhood of each point, and remove those points which do not meet a certain criteria. PCL’s StatisticalOutlierRemoval filter is an example of one such filtering technique. For each point in the point cloud, it computes the distance to all of its neighbors, and then calculates a mean distance.

By assuming a Gaussian distribution, all points whose mean distances are outside of an interval defined by the global distances mean+standard deviation are considered to be outliers and removed from the point cloud.

For illustration, take a look at the below image
![Statistical Oulier Filter ](https://user-images.githubusercontent.com/35863175/57986979-829f2f00-7a30-11e9-868c-5253be668f4a.png)

In code I have implemented it as follows 
![Statistical Outlier Filtering](https://user-images.githubusercontent.com/35863175/57986995-abbfbf80-7a30-11e9-8427-d305d57eab20.png)

## Clustering for Segmentation

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Algorithm

This algorithm is a nice alternative to k-means when you we know how many clusters to expect in our data, but we do know something about how the points should be clustered in terms of density (distance between points in a cluster).

The DBSCAN algorithm creates clusters by grouping data points that are within some threshold distance d from the nearest other point in the data.

The algorithm is sometimes also called “Euclidean Clustering”, because the decision of whether to place a point in a particular cluster is based upon the “Euclidean distance” between that point and other cluster members.

We can think of Euclidean distance the length of a line connecting two points, but the coordinates defining the positions of points in our data need not be spatial coordinates. They could be defined in color space, for example, or in any other feature space of the data.

In order to perform Euclidean Clustering, you must first construct a [k-d tree](http://pointclouds.org/documentation/tutorials/kdtree_search.php) from the cloud_objects point cloud.

The k-d tree data structure is used in the Euclidian Clustering algorithm to decrease the computational burden of searching for neighboring points.

Here I have implemented this technique in code
![Euclidean Clustering Code](https://user-images.githubusercontent.com/35863175/57987103-efff8f80-7a31-11e9-8645-f1eb1c77e842.png)

And for Visualization of the clusters, here I have done it as follows
![Visualizing Euclidean Clustering](https://user-images.githubusercontent.com/35863175/57987143-508ecc80-7a32-11e9-96ea-67f2a07ed2d7.png)

And this image shows us the clustered segmented image after the application of DBSCAN or Euclidean clustering
![PCL_Segmented](https://user-images.githubusercontent.com/35863175/57987126-23dab500-7a32-11e9-8f7a-f05653883b0b.png)

## Object Recognition

To perform successful object recognition mechanism, we need to extract the useful properties of the object which will help the Machine Learning algorithm to distinguish the various kinds of objects, these properties are called features. For example in our case these are Color and Shape of the objects. 

The below given functions compute color histograms and compute normal histograms can be found in **features.py** file inside the **Python Code Files** folder.

### Computation of Color Histograms
To extract the color features, there is concept of histograms which enables us to extract individual color spaces as features. We can use either RGB (Red, Green and Blue) color space and extract the R, G and B components as histograms separately or we can also use HSV (Hue, Saturation and Value) color space which is much more robust than RGB because lighting conditions do not affect HSV.

In my code I have provided both the options for RGB and HSV selection as follows for **computing color histograms**

![Compute Color Histograms 1](https://user-images.githubusercontent.com/35863175/57999876-d2f7aa80-7a8b-11e9-870c-4d3d5be95cb3.png)

![Compute Color Histograms 2](https://user-images.githubusercontent.com/35863175/57999936-063a3980-7a8c-11e9-92a9-147852c5d057.png)

### Computation of Normal Histograms
To extract the shape information, we use the concept of surface normals, which simply means drawing a perpendicuar line from the surface of the object going away from the object. These collection of surface normals are very efficiently able to determine the shape information and we can use these to **compute normal histograms** similar to that of color histograms done above.

Here is the code for normal histograms

![Compute Normal Histograms](https://user-images.githubusercontent.com/35863175/58000069-8791cc00-7a8c-11e9-8bdc-83b24b62e057.png)

### Support Vector Machine 
Support Vector Machine or "SVM" is just a funny name for a particular supervised machine learning algorithm that allows us to characterize the parameter space of your dataset into discrete classes. SVMs work by applying an iterative method to a training dataset, where each item in the training set is characterized by a feature vector and a label. Applying an SVM to this training set allows you to characterize the entire parameter space into discrete classes. The divisions between classes in parameter space are known as "decision boundaries".

In the **Python Code Files** folder, you will find **capture_features.py** file, which I have used to extract the features of the training set, and the features are saved finally as **training_set.sav** file. In the **capture_features.py**, you can also observe that models list, I have changed according to the objects presnt in World 1, World 2 and World 3.

This **training_set.sav** file is used to train our SVM by calling the code **train_svm.py**. This can also be found in the **Python Code Files** folder. For example, the below code shows that I have used the file as follows

![Trainnig Set Load](https://user-images.githubusercontent.com/35863175/58000778-9d07f580-7a8e-11e9-9548-0517e0ae2a8a.png)

I have used a linear SVM to train my model as it is computationally faster, but other methods of SVM can also be tried which are non linear and are able to model more complex features such as using RBF kernel etc. 

Finally, after training, I have saved my model as **model.sav** file. Here is the code snippet for that - 

![Training Model Saved Code](https://user-images.githubusercontent.com/35863175/58001011-2d463a80-7a8f-11e9-8513-dcd73cfc7af0.png)

**Also to mention specially, only this single generic SVM model has been used to predict the objects in each of the World 1, 2 and 3. No dedicated SVM for each World was used here**

After the training process, I proceeded to complete the object recognition code as follows - 

![Detecting Objects 1](https://user-images.githubusercontent.com/35863175/58001376-24a23400-7a90-11e9-8280-d885173d4386.png)

![Detecting Objects 2](https://user-images.githubusercontent.com/35863175/58001396-34ba1380-7a90-11e9-81c7-1cee17dedfa9.png)

The code is neatly commented with proper explanation.

## Pick and Place Setup
Now, I will start defining the code for the Pick and Place operation as well as the Publishing of the Detected Objects in Rviz by using the ROS

Here, I first, initialize the variables for the PR2 Mover arm - 

![PR2 Mover Initialize Variables](https://user-images.githubusercontent.com/35863175/58001711-228ca500-7a91-11e9-972f-10d47999bcf5.png)

Now, I calculate the centroids of the detected objects for a successful pick and place operation - 

![PR2 Mover Centroids of Detected Objects](https://user-images.githubusercontent.com/35863175/58001747-3afcbf80-7a91-11e9-818d-f43d5fee835a.png)

I intialize the loop which loops over all pick list objects and the PR2 performs the pick and place operation and finally save the detected objects with and pick and place positions in a YAML file. I perform this operation on three different Worlds each having different objects and save them as **output_1.yaml, output_2.yaml and output_3.yaml** files.

![PR2 Loop 1](https://user-images.githubusercontent.com/35863175/58001889-a777be80-7a91-11e9-8599-cf325ee9c0c4.png)

![PR2 Loop 2](https://user-images.githubusercontent.com/35863175/58001909-b52d4400-7a91-11e9-9bd4-43812fa5efb1.png)

Now, we look into the **ROS commands** - 

First, I initialize the ROS node

![ROS Node Initialization](https://user-images.githubusercontent.com/35863175/58002106-4997a680-7a92-11e9-8cd3-d02a88dae692.png)

Then, I created the ROS subscriber to subscribe to camera data point which is our point cloud data - 

![Create Subscribers](https://user-images.githubusercontent.com/35863175/58002198-90859c00-7a92-11e9-9590-a13d5431de03.png)

Now, I create the Publishers to publish the outputs on Rviz

![Create Publishers](https://user-images.githubusercontent.com/35863175/58002285-c9be0c00-7a92-11e9-8cc1-81acb69ac9fa.png)

Then, I load the trained model **model.sav** file as follows 

![Load Model from Disk](https://user-images.githubusercontent.com/35863175/58002393-14d81f00-7a93-11e9-8ceb-fe4b5d2b5365.png)

## Results 

### World 1 - test1.world

Normalized Confusion Matrix 

![World_1_CM](https://user-images.githubusercontent.com/35863175/58002468-58cb2400-7a93-11e9-9fd6-20736e295571.png)

Predicted Objects - 

![World_1_Predicted](https://user-images.githubusercontent.com/35863175/58002515-77c9b600-7a93-11e9-9c5d-1f5674ec8ccb.png)

### World 2 - test2.world

Normalized Confusion Matrix 

![World_2_CM](https://user-images.githubusercontent.com/35863175/58002570-916afd80-7a93-11e9-8746-e839e9d92202.png)

Predicted Objects - 

![World_2_Predicted](https://user-images.githubusercontent.com/35863175/58002604-a47dcd80-7a93-11e9-9de3-b700f00979d3.png)

### World 3 - test3.world

Normalized Confusion Matrix

![World_3_CM](https://user-images.githubusercontent.com/35863175/58002663-c6775000-7a93-11e9-8f36-990474986092.png)

Preicted Objects - 

![World_3_Predicted](https://user-images.githubusercontent.com/35863175/58002680-dabb4d00-7a93-11e9-8c46-2de0a988dca8.png)

### Future Work 

* A PR2 collision avoidance system can also be designed so that the robot does not collide with the surrounding objects while performing the pick and place operation.

* More sensors such as Lidar can be added to make the perception pipeline more robust rather than only using RGBD cameras.
