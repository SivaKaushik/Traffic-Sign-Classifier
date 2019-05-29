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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

In Data Images we have an exploratory visualization of the data set. It is shows images from the dataset randomly.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the number of maginitudes are decreased from 3 (RGB) to 1(gray) which will make computation easier for the CNN.
I didnt plot the images but looked at the shapes of the images after conversion by which i confirmed that the grayscaling has worked.
As a last step, I normalized the image data because as told in the course we want to keep the mean to zero and variance as low as possible so by standaridizing the images we keep the variance low and the model then eventually train itself and the varaince increases gradually.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:Keeping the mean = 0 and standard deviation to 0.1

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale and normalised image        | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|                                               |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                  |
| Convolution 5x5	    |1x1 stride,valid padding, output 10x10x16      |
|RELU   				|												|
| Max Pooling   		|2x2 strides, valid padding, output 5x5x16 .    |
| Flatten				| output 400 in 1D.                             |
|Drop out               |Probabilty of 0.5   							|
|Fully Connected Layer  |Output 120 in 1D                               |
|RELU                   |												| 
|Drop out               |												|
|Fully Connected Layer-2|output of 84 in 1D             				|
|RELU                	|												|
|Dropout                |												|
|Final Output           |   43 Unique Classes classifed 				|
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer for training the Neural Network
With Batch size of 128 and epoch number of 150 with a learning rate of 0.0004.
Added a couple of dropout layers to my CNN network to make the network more redundant.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 96% 
* test set accuracy of 94.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
--As it was used in the previous class in the coruse it proved an efficent starting point. The Achitecture is known as LeNet.

* What were some problems with the initial architecture?
--The accuracy wasnt changing from 89 with a lot more epochs it came it 90 but that wasnt enough.Too much information was lost.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
--I started by first tuning the batch size and epochs and then the learning rate without touching the network once i believed that the network has to become more redundant. I added dropout layer at couple of places and found that adding the dropout layer after flattening the and fullconnected circuit made it effect positive. It was overfitting at the beginning then it was fine. 

* Which parameters were tuned? How were they adjusted and why?
--I wanted the Algorithm to run more times in order for it to learn efficently so i first started by increasing the epochs and then adjusted the batch size few times but that wasnt effective. Then i tuned with the learning rate on the basis that slow learning rate can make the netwrok learn more effectively. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
--By using the dropout layer i got the best result. Preprocessing the image into a Grayscale and normalizing it.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five German traffic signs that I found on the web are stored in the Test_images folder with their respective labels corresponding to the signnames.

There is not special creteria for choosing them except for them to be very random while choosing,

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry Sign   		| No_entry sign   								| 
| speed limit 30		|Speed limit 30 								|
| Truck not allowed		| Truck not allowed								|
| turn right      		| Turn Right					 				|
| keep right			| Keep right         							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set which is at 94.9%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the laste cell of the Ipython notebook before the visualizing of CNN part.

For the first image, the model is very sure that this is a no entry sign (probability of 100), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| No Entry sign   									| 
| 1.000    				| speed limit 30 										|
| 1.000					| Trucks not allowed											|
| 1.000	      			| Turn right					 				|
| 1.000				    | stay right      							|


The CNN network did quiet a good job in recognizing these pictures even though the pictures were not bright enough and clear enough.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


