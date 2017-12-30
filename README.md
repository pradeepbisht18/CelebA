# CelebA
To determine whether a person in Image is wearing glasses or not and train our CNN Model.

In this assignment, your task is to implement a convolution neural network to determine whether the
person in a portrait image is wearing glasses or not. We will use the Celeb dataset which has more
than 200k celebrity images in total. In this task you are to use a publicly available convolutional
neural network package, train it on the Celeb images, tune hyperparameters and apply regularization
to improve performance.

Plan of Work

The following steps are the rough guidance and suggestions that you can follow to achieve better
evaluation performance on the problem. Some of the parameters in the steps are not pre-determined
(such as the resolutions you choose and the sizes of the dataset) so that certain variations should
appear among individuals/groups. We do not expect to see those working in different groups to pick
up exact same values.
Achieving all following steps requires a lot of work and is not required. You can choose some part
of it. As always, the more complete the experiments and project report are, the more preferable it is.
1. Extract feature values and labels from the data: Download the CelebA dataset from the
Internet and process the original data file into Numpy arrays that contains the feature vectors
and a Numpy array that contains the labels.
2. Reduce the resolution of the original images
3. Reduce the size of training set. For example, take out a small part of the entire set for
training.
2
4. Data Partition: Partition the picked-out CelebA dataset into a training set and a testing set.
You will use this partition and train your model on the training set.
5. Apply dropout or other regularization methods.
6. Train model parameter: For a given group of hyper-parameters such as dropout rates, the
number of layers and the number of nodes in each layer, train the model parameters on the
training set.
7. Tune hyper-parameters: Validate the classfication performance of your model on the valida-
tion set. Change your hyper-parameters and repeat the previous. Try to find what values those
hyper-parameters should take to give better performance on the testing set.
