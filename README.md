# DEEP_LEARNING_OCEAN
Deep learning Concepts from basic to advance 



Activation Function :
===================== 

In an artificial neural network, the activation function of a neuron defines the output of that neuron given a set of inputs.
 - Biologically inspired by activity in our brains, where different neurons fire, or are activated, by different stimuli.
 we have 
 * sigmoid ( ouptput ranges 0 - 1 ) -ive values tend to get close to zero and positive values are tend to get close to 1.
 * Relu ( -ive values are made equal to zero , concept here is more positive the value is more activated the neurons are)
   Rectified linear Unit
 * Softmax : for multiple categoires

Training the Model : 
====================
Training a netural neural network means optimizing the weights (within the model )of the neurons in the network to thier optimal value.
Training <==> optimization problem 
we use optimizer for the resolution of this problem : (common)
- SGD : stochastic Gradient Decent 
- (sgd objective : to minimise the loss function (which could be like mean square error or others) close to zero )
- ADAM 

- single pass of the data through the model is called an epoch
same data is passed in multile epoch for learning of the model.

- Process - 
============
for the first pass optimizer of the model randomly assign weights to the 
neurons and once first the pass is completed output is generated it is compared to the labeled data which was passed
the comparison bet the classified label and actual label of the data determine the error or the loss. at this point model computes 
the gradient (d(loss)/d(weight)) wrt each of the weight that has been set , once we have Gradient it is multiplied by 
the small value called learning rate (.001). The product of this multiplication (small value is updated as the new weight) and 
then multiple epochs are created for the same pass to minimise this error. 
These weights are continuosly get updated with each epoch.


Model to learn : 
================
Model will calculate loss at each input and what output it created after this input as the difference of correct label for the input value 
and predicted value, which is as such if model has to classify input into cat and dog and cat image is fed 
as input to the model which resembles 0.25 to cat (label -0) loss is calculated predicted lable - true label which is (.25 - 0) = 0.25
 
(predicted label - true label)

Loss funtion :
==============
mse each loss (predicted - actual ) is squared and avaergae is taken \.
Rmse. 
https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11

Learnng rate :
==============
Inorder to minimise the loss the optimizer calculate the gradient at the output of each epoch for each of the weight --> (d(loss)/d(weight)).
these gradients are then multilied with learning rate (lr) ranging from (.01 - .0001) we pass it as parameter but actual value may differ.
the lower side of the range (.0001) means we are taking small steps to find minimum in minimum loss function( minima) and it has higher chances to come accross the minimum loss 
point whereas higher  side (.01) of the range is bit risky as we take larger steps to find minima we might skip the point of minimum loss.
post determining the lr and the product we get from multiplication is relatively small values at each pass for each of the weights. 
these small values are then substracted from the respective weights to get the new weights for the next pass.

Train, test, validation data : 
=============================

Predicting with neural network :
=============================== 
perdicting on unseen data (test set)
predition using prediction = model.predict function 
where prediction will contain list of tuples for each of the sample test data and inside tuples thier 
respective probablities for each of the classes in the output.

Overfitting : 
============
when model learns most of the training data such that it performs well on the train data but fails on test data.
(it unable to generalise the data if the data slightly deviates from the training data) 
we can check overfitting with the help of the metric we pass while the validation step. we get loss and accuracy matrice when we train the model. 
if the validation matrix is conderably worse than the training matrix or test matrix is worse than training matrix then its overfitting. 

resolution : 1) add more training data / more data more diversity will be added.
2) use data augumentation (creating data with present data using actions like cropping, rotating, flipping,zooming)
3) something else we can do to reduce overfitting is to remove few layers or decrase the number of neurons from the model.
Basically reducing complexity of the model 
4) Dropout : if you add it in your model it will randomly ignore some set of nodes while training the model to generalise it.

Underfitting : 
==============

when the model is not able to classify the data that its has been trained on.

resoltion : 1)model is too simple increase the complexity by adding layers and neurons in each layer, what type of layer to use and where.
2) add more features in the input training data.
3) reduce drouput. 

Supervised learning : 
=====================
" the labeled data " input data is already labled and labels are used as classes.
 
Unsupervised leadrning :
======================== 
" unlabel data " : 
* no accuray metric as no labels are present.
* creating some structures and mapping those to the some outputs.
* clutering algos
* Autoencoders : it is an artificial neural network which takes input and outputs the reconstruction of this 
input. ex if we pass images of hand written digits to an autoencoder it reads the image and create a reconstruted version of it 
loss is determine by how closely similar this reconstructed version is to real iamge. 
usage : to denoise the image and extracting exact digit from the image. 

semisupervised Learning :
======================== 
using of label and unlabel data for training model. 
let say we have a large group of unlabled data, what we can do is labels few of the data and then train the model on this data 
using old supervised learning technique on the unlabeled data as input to the traised model. 
this is callled pesudo -labeling. 
now we have all data labeled which can be used as training set. 
usage : in order to label large data sets not manualy .  

Data Augumentation : 
====================
1) we want to add data to our training sets for smaples to overcome overfitting. 
2) cropping, zooming flipping rotating,inverting colors.

One hot encodings : 
==================
it set vector [x,x,x] for 3 classes and put 1 for each as label to the data, results are binary rather than ordinal. 
it adds up to the feature space and may result into curse of dimentionality.
LabelEncoder can turn [dog,cat,dog,mouse,cat] into [1,2,1,3,2], but then the imposed ordinality means that the average of dog and mouse is cat. 

CNN (convolutional neural network): 
-----------------------------------
these kind of neural network are specialised in picking out or detect partern. (better than MLP muli-layer perceptorn) 
how CNN is different than MLP ?
- Cnn has convolutional layers as hidden layers - performs convolutinal operations
- helps in detecting partern using filters 
paterns ? - multiple shapes,edges, texture,objects(eyes, hair, beaks etc). 
some filters may detect edges , some objects others shapes() in the image.(geometric filters at start of the network)
filters - matrix (having rows and columns - values within the matrix initialized randomly.)
deeper the networks goes the more sophisticated these filters become .
(some may detect eye, hair,face,fur,beaks in the later layers.)
filter slides on the image (pixel) and convolve entire input and put its dot product with image pixels as output.

Imnist dataset :

check out the link below for undertanding on the full operation of how filter works in background.

https://www.youtube.com/watch?v=vJiZqZRkIg8&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU&index=20 

Visualistion of CNNs :
=============================

VGG16 (Convolutional neural Network architecture) - Won Imagenet competition in 2014.it is an excellent vision model.
*   VGG16(also called -OxfordNet)name after Visual Geometry Group from Oxford .
*   Inception and ResNet has outperformed it.

gradient decent (minimizing the loss) vs gradient accent (maximising the loss) 
it is just to improve filter works (to activate it as much as possible in odrder to visually inspect the image)
 

Zero- Padding : 

28X28 pixel marix is convolve by filter of 3X3 in 26X26 possible positions hence the resultant would be 26X26 matrix pixels.
we can define it by a simple formula 
for input image of NXN pixels and FXF filter 
Output_size is given by - (N-F + 1) * (N-F +1)  so for 4X4 image and filter of 3X3 output is 2X2. 
to prevent information loss as here in the input image of 4X4 we can introduce padding. 
Padding : 2 types - 
1)VALID - No padding (input size is not maintainted)
2)SAME - same as to maintain the input size with the filter convolution. 
Padding is to add layer of pixels (intensity- 0) around the image so that it produces the output of the same size as the input size.


MAX-POOLING in CNN : 
* added in the Network after individual convolutinal layers, 
it reduces the dimentionality of image (output from previous convolution)by reducing the number of the pixels in the image. 

suppose the filter size is 2X2 for max pooling, this filter will move around the whole image 
with the mention stride and produce the output as max intensity of that convolve block,
* input from the previous layer is 26X26
 
https://www.youtube.com/watch?v=ZjM_XQa5s6s&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU&index=23

BACKPROPAGATION : 

Summary of the BP : 
1) Pass data to model via forward propagation.
2) Calculate loss on output
3) SGD minimizes the loss - by calculating the gradient of the loss function and updating weights 
    - Gradient is calculated via backpropagation.
Math behind it : 





