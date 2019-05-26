# Follow Me - Deep Learning

## Ouput Video!!!

https://youtu.be/XxCOg_dbuwY

## Major Issue

* An issue which I faced after running the command `python follower.py model_weights_4.h5` is as following -  `Fatal Python error: PyThreadState_Get: no current thread`. This issue is very common in Windows environment and I faced this too because I was using Windows 8.1. 

To resolve this, inside our activated environment provided by Udacity called RoboND, type the command - `conda update --all` . As sonn as I did this, the issue was resolved and my model worked! I was then able to capture a small video also!


## Future Enhancements

* I have used the Udacity provided data set for my training of the FCN, but more data can be collected and trained to achieve better accuracy.

* Skip connections can also be added to improve our result which will be based on certain number of connections, based on our number of layers we have.

* Adding more layers in our neural network will help our FCN model to capture more underlying features of our data set.
