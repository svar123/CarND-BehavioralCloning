
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/aaa.png "Model Architecture"
[image2]: ./images/Imagecenter0.PNG "Center Image" 
[image3]: ./images/Imageleft2.PNG "Left Image" 
[image4]: ./images/Imageright4.PNG "Right Image" 
[image5]: ./images/Imagecenter3.PNG "C Image"
[image6]: ./images/ImageFlippedcenter3.PNG "Flipped Image"
[image7]: ./images/Imagecenter0.PNG "Center Image"
[image8]: ./images/ImageAugmentedcenter0.PNG "Brightness Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 16 and 32 (model.py lines 112-116) 

The model includes RELU layers to introduce nonlinearity (code line 112,115), and the data is normalized in the model using a Keras lambda layer (code line 105). 

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, I used MaxPooling and also trained using large amounts of data (50,000 images) which were augmented and flipped(model.py lines 66-72). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the Udacity data set.

For details about how I augmented the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model (one convolutional layer and one fully connected layer) and add layers as needed.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because the dataset I used (Udacity dataset) was small. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a maxpooling layer. The mean squared error value on the validation set got better. I then added a second convolution layer with maxpooling and two more fully connected layers like the LeNet architecture.

To make training the model faster, I decided to scale the images to 64x64x3 from 160x320x3. This really helped the model train in less than 20 minutes rather than an hour or more.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I decided to use more images for training by augmenting and flipping the images from the three cameras(center, left and right) instead of just the center camera.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 112-124) consisted of a convolution neural network with the following layers and layer sizes.

Conv1 -> Relu -> MaxPooling -> Conv2 -> Relu -> MaxPooling -> Flatten ->
FullyConnected -> FullyConnected -> FullyConnected

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Here is a sample of images from the Udacity data set.

Center Camera Image:

![alt text][image2]

Left Camera Image:

![alt text][image3]

Right Camera Image:

![alt text][image4]


To augment the data set, I also flipped images and angles, and brightness thinking that this would generalize the data set. For example, here are a few images that have been flipped and brightness augmented:

Image:

![alt text][image5]

Image Flipped:

![alt text][image6]

Image:

![alt text][image7]

Image Brightness Augmented:

![alt text][image8]



I had a total of 8036 data points. I randomly shuffled the data set and put 20% of the data into a validation set. I then pre-processed the data by normalizing, cropping and scaling to 64x64x3 sized images. The images were randomly flipped and brightness augmented in the generator on the fly. I used 50000 images for training the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by mean square error values of the training and validation sets. I used an adam optimizer so that manually training the learning rate wasn't necessary.

