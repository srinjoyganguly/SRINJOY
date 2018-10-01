# FACE RECOGNITION AND SMILE DETECTION

![smile_face_detection](https://user-images.githubusercontent.com/35863175/46254204-2d1a1000-c4a9-11e8-89ef-0e9e79a9a123.jpg)

In this project, we are going to build a face recognition and smile detection system, which will be able to detect our faces, eyes as well as our smiles whenever we smile. The most important algorithm used for this technique is known as the Viola-Jones Algorithm which was created by Paul Viola and Michael Jones in 2001. This algorithm was designed to look only for features of frontal faces and is not applicable for side, up or down faces. It converts the image into black and white picture and starts scanning the whole image with a box (which can be scaled). So, this Viola-Jones tries to look for Haar Like features in our faces which are derived from the Haar wavelet. A photo may have these features especially if it's a face and so the algorithm detects these features to confirm if the image contains a face or not.  
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
![1](https://user-images.githubusercontent.com/35863175/46276367-7471d080-c57d-11e8-81d6-90452b48b469.JPG)
* Line 4 - We import the OpenCV library, which is used for working with computer vision projects and applications.
* Lines 7 to 9 - We load the cascades for the face, eyes and the smiles and these will be used to detect the facial features.
* Line 12 - We define a function that will do the face detection.
* Line 13 - It returns the tuples of 4 elements - x and y coordinates of the upper left corner of the rectangle which will detect the face and width and height of these rectangles. Gray is chosen as cascading works only with black and white images, 1.3 is the size of image reduction and 5 is the minimum number of neighbouring zones pixels which is required to accepted in order for a pixel zone to get accepted.
* Line 14 - We start the for loop which will iterate through each face, draw the rectangle around them and detect the face.
* Line 15 - This is for drawing the rectangles around the faces, where frame is image in which we want to draw our rectangle, (x,y) are coordinates of the upper left corner of rectangle, (x+w,y+h) are coordinates of lower right corner of rectangle, (255,0,0) is for color of the rectangle detecting the face and 2 is the thickness of the rectangle.
* Line 16 and 17 - These are the region of interest for the black and white and color images.
* Line 18 - Returns tuples of coordinates of upper left corner of rectangle and width and height of eyes for the detection of eyes.
* Line 19 and 20 - Iterating through the eyes, detecting them and drawing rectangle around the eyes.
* Lines 21 to 24 - The smile cascade is loaded and then iteration is carried out to look an detect for smiles. The rectangle is then drawn on the smiles and the fram is returned in the end, consisting of all the detections such as face, eyes and smiles.

![2](https://user-images.githubusercontent.com/35863175/46276372-79368480-c57d-11e8-8057-f0d56d913b48.JPG)
* Line 27 - This contains the last frame coming from the webcam and 0 is for webcam of computer (internal) and 1 will be for external webcam.
* Line 28 and 29 - A while loop which will iterate through the webcam frames indefinitely and will read the last frame from the webcam in which we are interested.
* Line 30 - Converts color image to balck and white (grayscale) image.
* Line 31 - We apply our detect function for the recognition of face and detection of eyes and smile in the webcam frame captured.
* Line 32 - Display preprocesses images in an animated way in a window.
* Line 33 and 34 - Used to break while loop when we press 'q' from the keyboard.
* Line 35 - Turn off the webcam.
* Line 36 - Destroys the windows inside which all the images were displayed.

## Outputs
![eye_face_detection](https://user-images.githubusercontent.com/35863175/46279671-69bc3900-c587-11e8-9071-efdf2a625b00.gif)
Eye and Face Detection


![face_smile_eye_detection](https://user-images.githubusercontent.com/35863175/46280081-8c028680-c588-11e8-8466-6fa8f6a3a86a.gif)
Face, Eye and Smile Detection

## Acknowledgements
* Udemy online platform for sustaining this beautiful course on AI.
* Huge ton of thanks to Hadelin De Ponteves and Kirill Ermenko for creating this wonderful course on AI
* Lots of thanks to Jordan Sauchuk (teaching asistant) for providing the appropriate instructions of installation on Windows and clearing doubts in the course.
* Style of explanation of the code is inspired from Adrian Rosebrock. His [Linkedin](https://www.linkedin.com/in/adrian-rosebrock-59b8732a) and [website](https://www.pyimagesearch.com/author/adrian/).
* Google Images


