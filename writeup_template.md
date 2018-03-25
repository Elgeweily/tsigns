# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/datasets_vis.jpg "Visualization"
[image2]: ./writeup/colorvsgray.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./writeup/im1.jpg "Traffic Sign 1"
[image5]: ./writeup/im2.jpg "Traffic Sign 2"
[image6]: ./writeup/im3.jpg "Traffic Sign 3"
[image7]: ./writeup/im4.jpg "Traffic Sign 4"
[image8]: ./writeup/im5.jpg "Traffic Sign 5"
[image9]: ./writeup/visualize_cnn_trained.jpg "Trained Visualization"
[image10]: ./writeup/visualize_cnn_untrained.jpg "Untrained Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the given data sets. The bar charts are showing the dataset distributions of each of the training, validation and testing datasets, in other words, how many examples of each type of sign are in each dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color has practically no effect on the accuracy of the classification, after all the shape and printing of the sign is what mostly determines it's class, if there were more than one sign class with the same shape and printing but different colors, then color would be very useful, but in our case, removing the color would make the network simpler and reduce the training time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data, i.e subtracted the mean, in order to center the data around zero, since the weight variables are randomly initialized with zero as the mean, so it makes sense that our data should have a zero mean as well, otherwise the weight variables will spend some of the training time just trying to find out where the mean of our data is, then divided by the standard deviation, which scales down the maximum value of any pixel to near unity, to make sure that bright images won't have a very high activation compared to darker images.

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a CNN consisting of 3 convolutional layers + 3 fully connected layers connected to the final output layer as described below:

| Layer         			|     Description	        									| 
|:-------------------------:|:-------------------------------------------------------------:| 
| Input						| 32x32x1 Grayscale Image										| 
| 1st Convolution 3x3		| 1x1 Stride, Valid Padding, Output 30x30x24					|
| RELU						|																|
| Max pooling 2x2			| 2x2 Stride, Valid Padding, Output 15x15x24					|
| 2nd Convolution 3x3		| 1x1 Stride, Valid Padding, Output 13x13x36					|
| RELU						|																|
| Max pooling 2x2			| 2x2 Stride, Valid Padding, Output 6x6x36						|
| 3rd Convolution 3x3		| 1x1 Stride, Valid Padding, Output 4x4x48						|
| RELU						|																|
| 1st Fully Connected Layer | Input 768 Flattened, Output 384, with 0.6 keep_prob Dropout	|
| RELU						|																|
| 2nd Fully Connected Layer | Output 192, with 0.6 keep_prob Dropout						|
| RELU						|																|
| 3rd Fully Connected Layer	| Output 96, with 0.6 keep_prob Dropout							|
| RELU						|																|
| Output Layer				| Output 43														|
| Softmax					|																|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer as a backpropagation algorithm, Mean of Cross Entropies as a loss function, softmax function as an Activation for the output layer, batch size of 128 (with data shuffling for each epoch), 10 epochs, learning rate of 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8% (in Cell [33])
* validation set accuracy of 97.1% (in Cell [33])
* test set accuracy of 95.2% (in Cell [34])

If an iterative approach was chosen:
* As a starting point I've used Lenet-5 CNN Architecture, which works very well for hand written digits, and relatively well for traffic signs, since traffic signs have printed digits and other abstract shapes and letters that resemble the lines and curves of digits.
* Using Lenet-5 I've arrived at approximately 89% accuracty, such low accuracy compared to the handwritten digits could be due to the fact that traffic signs have more complex figures (walking pedestrians, etc), different sign shapes (circular, rectangular, triangular), in addition to the fact that the images in the provided dataset were not very clear, some of the distortions present in the dataset are: blurred signs, very dark images, damaged signs, low/high saturation, angled signs, in addition to the existence of some objects in the background of the images that could have confused the network and resulted in incorrect classifications.
* The most noticeable shortcoming of the above model was overfitting, discovered when noticing that the accuracy when evaluating the model on the training dataset was much higher (about 20% higher) than the accuracy when evaluating using the validation dataset, which simply means that the model can't generalize what it learned from the training dataset on new images, due to the imperfections of the training images discussed above, for example when the model sees the same traffic sign but in different blur/lighting/angle, etc, it won't recognize it as being the same sign. To address this issue dropout was introduced to the fully connected layers (with 60% keep probability), dropout makes sure that the classification doesn't depend strongly on a single specific feature that might not always be present in the image (might be present in the training set but not in the test set), but instead to depend on a consensus of different features, making sure that these discovered features are inherent to the shape of the sign itself and not just a conicident feature (blur, etc).
* Using dropout the accuracy was increased to about 93.5% - 94%, as an attempt to further increase the model accuracy, the kernel size of the convolutional layers were reduced to 3x3 instead of 5x5, in order for the convolutional layers to be able to detect finer details, then the depth of the kernel (in the first conv. layer) was increased to 24 instead of 6, in order to discover more features in each image, then the number of convolutional layers were increased to 3 layers instead of 2 layers, to further abstract the images and increase their depth (number of detected features), finally an additional fully connected layer was added (3 fcs instead of 2 fcs), which makes the whole network deeper, increase it's non-linearity and it's ability to fit more complex data classes.
* CNNs are suitable for classifiying images because it solves the problem of having to connect every pixel in an image to every neuron in the network, resulting in a very high number of parameters specially in deeper (many layers) networks. Convolution solves this problem by connecting a group of adjacent pixels (instead of every pixel) to each of the network's neurons, in addition to that, the parameters are spatially shared, since we don't need different weights to classify the same object whether it is on the right, left, bottom or top of the image, we just need to scan the same weights kernel throughout the image, in order to find our object.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is shot at an angle, and thus the number "120" is skewed and is difficult to recognize, the second image is a bit dark, but is not difficult to classify, the third image has heavy shadows on the sign face, the sign in the fourth image is a bit worn and has lots of scratches, the sign in the last image is a bit angled, not centered and has some dirt and water droplets on it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h)	| Go straight or left							| 
| Priority Road			| Priority Road 								|
| Stop					| Turn left ahead								|
| Road Work				| Road Work					 					|
| Keep Right			| Keep Right									|


The model was able to correctly guess 3 out of 5 traffic signs, which gives it an accuracy of 60%. this is quite far from the accuracy of the test dataset (95%), these 5 images were selected after choosing 5 typical images from the web, that were all classified correcly (100%), so I thought I need to choose difficult to classify images, that will give insight on the shortcomings of the model, so these 5 images did the job, showing that the model is not robust enough to skewed images (the first misclassification), since the images in the provided dataset are all shot from a distance so this issue was not apparent in the test dataset). For the second misclassification, apparently the shadows were heavy enough to confuse the classifier. It is worth noting that when the classifier is confused by distortion, the classification is not even close to the correct classification, which shows that "similar" from a human's point of view is not necessarily "similar" from a machine's point of view, and vice versa, and that the neural network finds it's own criteria for classification which are not necessarily the same criteria used by humans. Apparently the machine thinks that a "120km/h" sign (when distorted enough) looks similar to a "Go straight or left sign"!, which would never be the case from a human's point of view. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For the first image, the model is quite sure that this is a Go Straight or Left Sign (probability of 0.97), while the image actually contains a Speed limit (120km/h) sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97					| Go Straight or Left							| 
| .0247					| No Entry 										|
| .00260				| Roundabout Mandatory							|
| .00041				| Speed Limit (20km/h)			 				|
| .00015				| Speed Limit (30km/h)							|

For the second image, the model is almost 100% sure that this is a Priority Road Sign, which is the correct classification. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ≈1					| Priority Road   								| 
| e-13					| Roundabout Mandatory 							|
| e-14					| No Vehicles									|
| e-14					| Yield			 								|
| e-14					| Speed Limit (120km/h)							|

For the third image, the model is quite sure that this is a Turn Left Ahead Sign (probability of 0.73), while the image actually contains a Stop Sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .73					| Turn Left Ahead								| 
| .1189					| Ahead Only									|
| .04983				| Yield											|
| .01388				| Priority Road									|
| .01272				| Go Straight or Right							|

For the fourth image, the model is quite sure that this is a Road Work Sign (probability of 0.98), which is the correct classification. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98283				| Road Work										| 
| .0154					| Wild Animals Crossing							|
| .001468				| Bumpy Road									|
| .0001728				| Dangerous Curve to the Right					|
| .00009314				| Bicycles Crossing								|

For the fifth image, the model is almost 100% sure that this is a Keep Right Sign, which is the correct classification. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ≈1					| Keep Right									| 
| e-18					| Go Straight or Right							|
| e-20					| Dangerous Curve to the Right					|
| e-25					| No Entry										|
| e-25					| Road Work										|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below are the feature maps visualizations of the first convolutional layer for the trained network vs the untrained network, it is noticeable that in the untrained maps, some of the backgrounds have high activations, which is expected since the untrained network has no idea what it's looking for, and doesn't know that the backgrounds are totally irrelevant for the classification, on the other hand, the trained network has learned that, and has mostly blacked out the backgrounds, effictively ignoring them. Also we can see that the trained network has strong activations for the edges of the sign and it's internal shapes, which is not the case with the untrained network.

### Trained Network Visualization

![alt text][image9]

### Untrained Network Visualization

![alt text][image10]


