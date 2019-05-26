[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Follow Me - Deep Learning

In this project, we will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

<img width="797" alt="sim_screenshot" src="https://user-images.githubusercontent.com/35863175/58378540-1afa5e00-7fb3-11e9-9106-60005119da1a.png">

## Installation Instructions and Project Setup Details
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies - Software**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

**Hardware**

I have used the Udacity GPU Workspcae to train my model because I didn't had any any access to any powerful GPU.

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 


## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py model_weights_4.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`

**The [Rubric Points](https://review.udacity.com/#!/rubrics/1155/view) are Explained below in detail**

## Convolutional Neural Network

Before we dive deep into the the underpinnings of a Fully Convolutional Network (FCN), first let us understand what is a Convolutional Neural Network (CNN), as this will help us to understand the FCN's in abetter manner.

The first step for a CNN is to break up the image into smaller pieces. We do this by selecting a width and height that defines a filter.

The filter looks at small pieces, or patches, of the image. These patches are the same size as the filter. 

![breaking up an image](https://user-images.githubusercontent.com/35863175/58379566-f0180600-7fc2-11e9-8722-f1b2198fc8c1.png)

We then simply slide this filter horizontally or vertically to focus on a different piece of the image.

The amount by which the filter slides is referred to as the 'stride'. The stride is a hyperparameter which you, the engineer, can tune. Increasing the stride reduces the size of your model by reducing the number of total patches each layer observes. However, this usually comes with a reduction in accuracy.

So, while doing these convolutions, we are grouping together adjacent pixels and treating them as a collective. This helps our CNN to classify local patterns, like shapes and objects, in an image.

**Filter Depth**

It's common to have more than one filter. Different filters pick up different qualities of a patch. For example, one filter might look for a particular color, while another might look for a kind of object of a specific shape. The amount of filters in a convolutional layer is called the filter depth.

![filter depth](https://user-images.githubusercontent.com/35863175/58379615-82200e80-7fc3-11e9-89ac-44c57a4ae552.png)

If we have a depth of k, we connect each patch of pixels to k neurons in the next layer. This gives us the height of k in the next layer, as shown below. In practice, k is a hyperparameter we tune, and most CNNs tend to pick the same starting values.

<img width="303" alt="filter-depth-2" src="https://user-images.githubusercontent.com/35863175/58379626-bd224200-7fc3-11e9-8f64-b54f24fc2939.png">


Multiple neurons can be useful because a patch can have multiple interesting characteristics that we want to capture. Having multiple neurons for a given patch ensures that our CNN can learn to capture whatever characteristics the CNN learns are important.

Typically, a CNN is followed by a fully connected layer and then by a softmax for predictions.

Now, since we know what a CNN is, we can now proceed towards Fully Convolutional Networks (FCN)

## Fully Convolutional Networks (FCN)

As we saw in CNN, that it is followed by a simple fully connected layer which is just a stack of neurons and this layer actually does not  capture the spatial information present in the images. So, if we can convert this fully connected layer to a fully convolutional network then they will be able to preserve the spatial information throughout the entire network!

Now, FCN take advantage of three special techniques, here given as follows - 

1. Replace fully connected layers with 1x1 convolutional layers, as shown below - 

![1x1 convolution](https://user-images.githubusercontent.com/35863175/58379955-21df9b80-7fc8-11e9-833d-4b9803b4cd7b.PNG)

2. Upsampling through the use of Transposed convolutional layers, as shown - 

![upsampled](https://user-images.githubusercontent.com/35863175/58379951-12f8e900-7fc8-11e9-8662-956715dfaa0f.PNG)

3. Skip Connections, as shown below - 

![skip_connections](https://user-images.githubusercontent.com/35863175/58379945-0379a000-7fc8-11e9-9d6a-ccabe31828d5.PNG)

These skip connections allow the network to use information from multiple resolution scales. I have not used skip connections for my model, but to make the model more robust, it can be used. I have mentioned this in the Future Enhance ments section.

Structurally, a FCN is usually comprised of two parts - **Encoder** and **Decoder** as shown below - 

![encoder_decoder](https://user-images.githubusercontent.com/35863175/58379957-315ee480-7fc8-11e9-99a0-7882af5dfad5.PNG)

### Encoder 

**Encoder** - The encoder is a series of convolutional layers such as VGG or ResNet. The goal of the encoder is to extract features from the image. 

* **A Disadvantage of Encoding or Downsampling** - Since each time we convolve, we are **reducing the size of our image**, which means we are **losing important image information** and only **considering the small picture elements only and eventually losing out the bigger picture elements.** To solve this issue we use **skip connections**, which is discussed after the Upsampling step.

**Decoder** - The decoder up scales the output of the encoder such that it's the same size as the original image. The decoder is explained in more detail after the **FCN points**

Now, I dive into little bit detailed discussion on each of the **FCN points** mentioned above - 

### 1x1 Convolutions 

* The 1x1 convolutions control the depth of the input volume as it is passed to the next layer, either decrease it, or increase it, or just add a non-linearity when it doesn’t alter the depth. This control is achieved by the choosing the appropriate number of filters. We can control the other two dimensions - width and height by the filter sizes and padding parameters, or use pooling to reduce width and height. This also capyures the global context of the scene.

* In the case when it is reduces the dimensions, it is a means to reduce computations.

### Upsampling through Transposed Convolutional Layers

* Transposed Convolutions help in upsampling the previous layer to a desired resolution or dimension. Suppose you have a 3x3 input and you wish to upsample that to the desired dimension of 6x6. The process involves multiplying each pixel of your input with a kernel or filter. If this filter was of size 5x5, the output of this operation will be a weighted kernel of size 5x5. This weighted kernel then defines your output layer. 

Here is an image to depict the above information in a more accurate manner - 

![transposed-conv](https://user-images.githubusercontent.com/35863175/58380134-1772d100-7fcb-11e9-8710-e7ea1de41f5c.png)

The 3x3 weighted kernel (product of input pixel with the 3x3 kernel) is depicted by the red and blue squares, which are separated by a stride of 2. The dotted square indicates the padding around the output. As the weighted kernel moves across, the stride determines the final dimension of the output. Different values for these will result in different dimensions for the upsampled output. 

### Skip Connections 

* Skip connections are a way of retaining the information easily. To account for the disadvantage of 1x1 convolutions, in the skip coonections, the output of one layer is connected to a non adjacent layer. 

* Here as shown,  the output of the pooling layer from the encoder combine with the current layers output using element wise addition opeartion - 


![skip connections between enc and dec](https://user-images.githubusercontent.com/35863175/58380218-3de53c00-7fcc-11e9-87f1-53537cd0dcbb.PNG)

* These skip connections help us to make precise segmentation decisions, as shown below - 

![skip connections example](https://user-images.githubusercontent.com/35863175/58380284-ded3f700-7fcc-11e9-9312-89aea2d673c9.png)

### Encoder

#### Separable Convolutions

Separable Convolutions, introduced here, is a technique that reduces the number of parameters needed, thus increasing efficiency for the encoder network. 

Separable convolutions, also known as depthwise separable convolutions, comprise of a convolution performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer.

This is different than regular convolutions that we covered before, mainly because of the reduction in the number of parameters. Let's consider a simple example.

Suppose we have an input shape of 32x32x3. With the desired number of 9 output channels and filters (kernels) of shape 3x3x3. In the regular convolutions, the 3 input channels get traversed by the 9 kernels. This would result in a total of 9*3*3*3 features (ignoring biases). That's a total of 243 parameters.

In case of the separable convolutions, the 3 input channels get traversed with 1 kernel each. That gives us 27 parameters (3*3*3) and 3 feature maps. In the next step, these 3 feature maps get traversed by 9 1x1 convolutions each. That results in a total of 27 (9*3) parameters. That's a total of 54 (27 + 27) parameters! Way less than the 243 parameters we got above. And as the size of the layers or channels increases, the difference will be more noticeable.

The reduction in the parameters make separable convolutions quite efficient with improved runtime performance and are also, as a result, useful for mobile applications. They also have the added benefit of reducing overfitting to an extent, because of the fewer parameters.


### Decoder

* The **decoder up scales** the output of the encoder such that it's the same size as the original image.

* **Bilinear Upsampling** by a factor of 2 is generally used in the decoder blocks to recover resolution then add it together with the previous encoders layers outputs to get the required up-size. Different factors of upsampling can be used if required.

Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent.

Let's consider the scenario where you have 4 known pixel values, so essentially a 2x2 grayscale image. This image is required to be upsampled to a 4x4 image. The following image gives a better idea of this process.

![bilinear upsampling](https://user-images.githubusercontent.com/35863175/58380426-7ab23280-7fce-11e9-9780-2621c36c4cc3.png)



The unmarked pixels shown in the 4x4 illustration above are essentially whitespace. The bilinear upsampling method will try to fill out all the remaining pixel values via interpolation. Consider the case of P5 to understand this algorithm.

We initially calculate the pixel values at P12 and P34 using linear interpolation.

That gives us,

P12 = P1 + W1*(P2 - P1)/W2

and

P34 = P3 + W1*(P4 - P3)/W2

Using P12 and P34, we can obtain P5:

P5 = P12 + H1*(P34 - P12)/H2

For simplicity's sake, we assume that H2 = W2 = 1

After substituting for P34 and P12 the final equation for the pixel value of P5 is:

P5 = P1*(1 - W1)*(1 - H1) + P2*W1*(1 - H1) + P3*H1*(1 - W1) + P4*W1*H1

While the math becomes more complex, the above technique can be extended to RGB images as well.

### Batch Normalization 

* **Batch normalization** is also used in each FCN layer and it is based on the idea that, instead of just normalizing the inputs to the network, we normalize the inputs to layers within the network. It's called "batch" normalization because during training, we normalize each layer's inputs by using the mean and variance of the values in the current mini-batch. Batch normalization presents us with few advantages: Networks train faster, higher learning rates,Simplifies the creation of deeper networks, and provides a bit of regularization.

**Advantages of Batch Normalization are as follows**

1. **Networks train faster** – Each training iteration will actually be slower because of the extra calculations during the forward pass. However, it should converge much more quickly, so training should be faster overall.

2. **Allows higher learning rates** – Gradient descent usually requires small learning rates for the network to converge. And as networks get deeper, their gradients get smaller during back propagation so they require even more iterations. Using batch normalization allows us to use much higher learning rates, which further increases the speed at which networks train.

3. **Simplifies the creation of deeper networks** – Because of the above reasons, it is easier to build and faster to train deeper neural networks when using batch normalization. 

4. **Provides a bit of regularization** – Batch normalization adds a little noise to your network. In some cases, such as in Inception modules, batch normalization has been shown to work as well as dropout. 

## Ouput Video!!!

https://youtu.be/XxCOg_dbuwY

## Major Issue

* An issue which I faced after running the command `python follower.py model_weights_4.h5` is as following -  `Fatal Python error: PyThreadState_Get: no current thread`. This issue is very common in Windows environment and I faced this too because I was using Windows 8.1. 

To resolve this, inside our activated environment provided by Udacity called RoboND, type the command - `conda update --all` . As sonn as I did this, the issue was resolved and my model worked! I was then able to capture a small video also!


## Future Enhancements

* I have used the Udacity provided data set for my training of the FCN, but more data can be collected and trained to achieve better accuracy.

* Skip connections can also be added to improve our result which will be based on certain number of connections, based on our number of layers we have.

* Adding more layers in our neural network will help our FCN model to capture more underlying features of our data set.
