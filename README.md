
# Behavioral Cloning

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

## Submission Details

I created a CNN using Keras that takes as input a 160x320 pixel image and predicts a steering angle. The trained model is able to drive both tracks smoothly at 9mph. My submission includes:
- model.py -> script to create and train the model
- drive.py -> file for driving the car in autonomous mode
- model.hdf5 -> my trained model
- Track1_final.mp4 -> video of the car driving the first track
- Track2_final.mp4 -> video of the car driving the second track

[//]: # (Image References)

[image1]: ./inputimg.png "Input"
[image2]: ./cropping.png "Cropping"
[image3]: ./pooling.png "Pooling"


## Model Architecture and Training Strategy

The model.py file contains the details how I created and trained my CNN.


### Model Architecture
I tried out many different architectures and ended with the following high level description:

Layer | Output shape | Number of Parameters
------------ | -- | -- |
INPUT | (batch, 160, 320, 3) | 0 |
CROPPING | (batch, 60, 320, 3) | 0 |
NORMALIZE | (batch, 60, 320, 3)| 0 |
MINPOOL | (batch, 30, 160, 3) | 0 |
MINPOOL | (batch, 30, 40, 3) | 0 |
CONV(1x1)/CONV(3x3)/CONV(5x5) | (batch, 30, 40, 60) | 2.040 |
MAXPOOL | (batch, 15, 20, 60) | 0 |
BATCHNORM | (batch, 15, 20 , 60) | 240 |
CONV | (batch, 15, 20, 20) | 10.820 |
MAXPOOL | (batch, 7, 10, 20) | 0 |
BATCHNORM | (batch, 7, 10, 20) | 80 |
CONV | (batch, 7, 10, 10) | 1.810 |
FLATTEN | (batch, 700) | 0 |
DENSE | (batch, 50) | 35.050 |
DENSE | (batch, 30) | 1.530 |
OUTPUT | (batch, 1) | 31 |

Total number of parameters: 51.601

I used Relu-Units as non-linearities. My observation was that dropout layers did not lead to better models. So, I prevented overfitting mainly by reducing the size of the network, ending up with the above architecture. For training, I used the Adam optimizer.


### Training Data

I obtained training data by driving in the simulator. The car had three cameras. I ended up with 151.570 images from the center camera (two-third of the data is from the first track). Using cross-validation to determine a correction of the steering angle, I could also use the images from both the other cameras. Also, I flipped each image to obtain more training data and prevent any right-left steering bias. In total, this lead to 454.704 images for training and 113.682 images for validation.

In total, I drove both tracks in both directions and added more data of the car steering back to the middle when starting at a side lane.


##  Solution Approach

I used 80% of the data for training and 20% for validation. I started with a pretty big model (lots of paramters, size ca. 50MB) but soon noticed that larger models take much longer to train without obtaining better validation errors. Thus, I ended up with a much smaller model (size ca. 660 KB).

Example of an input image:
![alt text][image1]


Since we do not need the information of the top part of an image and the bottom part of an image, I used a cropping layer first. To decide which kind of cropping I used the following figure and ended up with 80 pixels from the top and 20 from the bottom.

![alt text][image2]

In order to reduce the input size, I applied pooling layers. I decided to use a 2x2 MinPooling layer followed by a 1x4 MinPooling layer. The 1x4 filter was used since the road is wide and contains many unnecessary information. Examples of different pooling and cropping combinations can be seen here:

![alt text][image3]

After a normalizing layer, I used an inception kind of structure where I applied a 1x1, a 3x3 and a 5x5 convolution to the input and merge these filters together. Then, I used two more 3x3 convolutional layers, followed by two fully connected layers and a final single neuron as output.

I only trained the model for one epoch, i.e., I used early stopping to prevent further overfitting.


## Data generator

Since the dataset is very large, I used a data generator to load the batches into the RAM. The flipping and the change of the steering angle for the side camera images was done in the generator routine.


# Simulation

The simulation videos are included. The car can drive autonomously through both tracks safely with a human-like driving behavior. It tries to stay in the middle of the road.
