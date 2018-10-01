# FACE RECOGNITION AND SMILE DETECTION

![smile_face_detection](https://user-images.githubusercontent.com/35863175/46254204-2d1a1000-c4a9-11e8-89ef-0e9e79a9a123.jpg)

In this project, we are going to build a face recognition and smile detection system, which will be able to detect our faces, eyes as well as our smiles whenever we smile. The most important algorithm used for this technique is known as the Viola-Jones Algorithm which was created by Paul Viola and Michael Jones in 2001. This algorithm was designed to look only for features of frontal faces and is not applicable for side, up or down faces. It converts the image into black and white picture and starts scanning the whole image with a box (which can be scaled). So, this Viola-Jones tries to look for Haar Like features in our faces which are derived from the Haar wavelet. A photo may have these features especiall if it's a face and so the algorithm detects these features to confirm if the image contains a face or not.  
![haar like features](https://user-images.githubusercontent.com/35863175/46273755-80a56000-c574-11e8-9ff3-f1b94285b4ca.png)

## Prerequisites For This Project
These are some of the prerequisites of the project which you have to be familiar with to enjoy learning about this project-
* Understanding of [Artificial Neural Networks](https://www.tutorialspoint.com/artificial_neural_network/) and [Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/). Some image processing knowledge is helpful, but not must for understanding the project
* Python programming experience with the knowledge of [Pytorch](https://pytorch.org/tutorials/) and [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_intro/py_intro.html).

## Installation Instructions
### For Windows, Mac and Linux
