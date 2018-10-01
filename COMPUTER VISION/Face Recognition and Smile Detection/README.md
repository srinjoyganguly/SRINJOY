# FACE RECOGNITION AND SMILE DETECTION

![smile_face_detection](https://user-images.githubusercontent.com/35863175/46254204-2d1a1000-c4a9-11e8-89ef-0e9e79a9a123.jpg)

In this project, we are going to build a face recognition and smile detection system, which will be able to detect our faces, eyes as well as our smiles whenever we smile. The most important algorithm used for this technique is known as the Viola-Jones Algorithm which was created by Paul Viola and Michael Jones in 2001. This algorithm was designed to look only for features of frontal faces and is not applicable for side, up or down faces. It converts the image into black and white picture and starts scanning the whole image with a box (which can be scaled). So, this Viola-Jones tries to look for Haar Like features in our faces which are derived from the Haar wavelet. A photo may have these features especiall if it's a face and so the algorithm detects these features to confirm if the image contains a face or not.  
![haar like features](https://user-images.githubusercontent.com/35863175/46273755-80a56000-c574-11e8-9ff3-f1b94285b4ca.png)

## Prerequisites For This Project
These are some of the prerequisites of the project which you have to be familiar with to enjoy learning about this project-
* Understanding of [Artificial Neural Networks](https://www.tutorialspoint.com/artificial_neural_network/) and [Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/). Some image processing knowledge is helpful, but not must for understanding the project
* Python programming experience with the knowledge of [Pytorch](https://pytorch.org/tutorials/) and [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_intro/py_intro.html).

## Installation Instructions
### For Mac and Linux
1. Install Anaconda software - Python 3.6 version from [here](https://www.anaconda.com/download/). This software comes with the Python IDE and lots of more libraries for data science and machine learning such as Numpy, pandas, matplotlib, scipy etc.
2. Open the terminal.
3. Download the Installations folder in your desktop given with this project.
4. Type the command - cd Desktop/Installations, and press enter to get inside the folder.
5. Now type the command - conda env create -f virtual_platform_linux.yml. Then press enter and it will install everything we need for this project. For mac, the file name should be virtual_platform_mac.yml.
6. Now open the Anaconda Navigator and in the Applications on tab select - virtual_platform.
7. Launch Spyder.

## For Windows
1. Install Anaconda software - Python 3.6 version from [here](https://www.anaconda.com/download/). This software comes with the Python IDE and lots of more libraries for data science and machine learning such as Numpy, pandas, matplotlib, scipy etc.
2. Download the Installations folder in your desktop given with this project.
3. Go to the installations folder and open virtual_platform_windows.yml file with notepad. In this file, delete the line 92 - pytorch = 0.1.2=py35_0.1.12cu80 and line 100 - torch==0.1.12. If you found the lines to be deleted already then skip this step.
4. Open the Anaconda Prompt which is similar to a terminal used in Linux or Mac and type the command - cd Desktop/Installations, and press enter to get inside the folder.
5. Now type the command - conda env create -f virtual_platform_windows.yml. Then press enter and it will install everything we need for this project.
6. Activate the virtual platform by typing and executing the command - activate virtual_platform.
7. Type the command - conda update --all.
8. Install Pytorch library by typing this command - conda install -c peterjc123 pytorch cuda80.
9. Install Torchvision by typing this command - pip install torchvision-0.2.0-py2.py3-none-any.whl.
10. Now open the Anaconda Navigator and in the Applications on tab select - virtual_platform.
11. Launch Spyder.

## Code Snippets with Detailed Explanation
### Elucidation of Happiness_Detector.py file
