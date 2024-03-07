# Handwritten digit recognition based on MNIST

## Introduction
The problem of handwritten digit recognition is a classic challenge in the field of machine learning and computer vision. It involves classifying images of handwritten digits into one of the ten classes, representing digits from 0 to 9. The significance of this problem lies in its wide range of applications, from postal mail sorting based on zip codes to bank check digit recognition.
To tackle this problem, we have chosen to use a neural network model trained on the MNIST dataset. The MNIST (Modified National Institute of Standards and Technology) dataset is a large database of handwritten digits that is commonly used for training and testing in the field of machine learning. It contains 60,000 training images and 10,000 testing images, each of which is a 28x28 grayscale image of a digit.
In the following sections, we will detail the steps to reproduce our project, present our code implementation, and analyze the results obtained. We hope that our work can contribute to the ongoing research in the field of handwritten digit recognition.

## Code
Using opencv2 and cmath
The project starts with preprocessing the MNIST data set, and use the vector feature and Opencv of C++ to preprocess the training set and test set.
This part of the code defines the vectors that store training and test data. train_images and test_images contain the image data, while train_labels and test_labels contain the corresponding labels.
This function is used to randomly assign initial values to the weight matrix and bias vector of the neural network so that it has a reasonable set of initial parameters at the beginning of training. This function initializes the weight matrix (w1) of the hidden layer and the weight matrix (w2) of the output layer respectively, with a random number in the range [-0.05-0.05], and the bias vector (bias1) of the hidden layer and output layer , bias2) for initialization, a random number in the range [-0.1-0.1]. This process ensures that the neural network avoids falling into local minima at the beginning of training and helps the model better learn the characteristics of the training data.

### Activation function - Sigmoid
 
The role of the activation function in the neural network is to introduce nonlinearity so that the network can learn and represent more complex functions. The Sigmoid function is particularly suitable for the output layer and is often used in binary classification problems. In the hidden layers of neural networks, it can introduce nonlinearity, allowing the network to learn complex features and patterns. In the output layer, it can map the output of the network to between 0 and 1 to facilitate probabilistic interpretation or decision-making for binary classification problems.

Obtain hidden layer output and forward propagation of neural network
In the forward propagation process of a neural network, obtaining the output of the hidden layer through the input image is a crucial step.

These two functions together complete the forward propagation process of the neural network. First, the get_hidden_out function calculates the mapping of the input image to the hidden layer through the weight matrix, and then obtains the output of the hidden layer through the Sigmoid activation function. Then, the get_z function uses the output of the hidden layer, again through the mapping of the weight matrix to the output layer, and obtains the final output of the neural network through the Sigmoid activation function.

## Detailed explanation of training process (code part omitted)
Loop structure:
Outer loop (for (e = 1; e <= epoch; e++)): controls the total number of rounds of training, that is, the number of epochs.
Inner loop (for (int im = 0; im < train_images.size(); im++)): Iterates through each image in the training set to update the parameters of the model.
Gradient calculation and weight update:
In each inner loop iteration, the loss value is first calculated, and then the gradient is calculated based on the loss value.
The gradients from the hidden layer to the output layer (w2) and the input layer to the hidden layer (w1) are calculated by backpropagation, as well as the corresponding bias term gradients.
Using the calculated gradients, the weight matrix and bias vector are updated via the gradient descent method to minimize the loss function.
Function call:
get_hidden_out(train_images[im]): Get the hidden layer output.
get_z(hidden_out): Get the final output of the neural network.
get_loss(z, train_labels[im]): Calculate the loss value.
test(): Perform model testing at the end of each epoch to evaluate the performance of the current model.

## Summary of training process
This code is responsible for training the neural network model and continuously adjusting the weights and biases through multiple iterations (epochs), so that the model gradually learns and optimizes its performance on the handwritten digit recognition task. Each iteration involves steps such as forward propagation, loss calculation, back propagation, and parameter update, thereby continuously improving the accuracy of the model. At the end of each epoch, evaluate the model's performance on the test set by calling the test() function.
After completing the iteration, the test function predicts each image in the test set and gets the output of the model. Compare the model's output to the true labels to determine whether the model's predictions for each image are accurate. Count the number of correct predictions and calculate the accuracy of the test set (the ratio of the number of correctly predicted images to the total number of images). Output the accuracy information of the current epoch to monitor the performance changes of the model.

## Experimental results
By observing the accuracy of each epoch, you can understand the learning trends and performance changes of the model during the training process. Ideally, as training proceeds, the model's accuracy should gradually improve, indicating that the model has gradually learned better features and patterns. And the overall accuracy increases as the number of epochs.

On average, an epoch takes 550 seconds. The program implements a neural network with a single hidden layer in a relatively concise way. The code structure is clear and the functions are modular, which facilitates subsequent expansion and modification. Using global variables for parameter configuration allows users to easily adjust hyperparameters such as learning rate and number of iterations.

## Summary
By setting hyperparameters such as learning rate and number of iterations in the code, users can optimize the performance of the model by adjusting these parameters. This reminds users of the importance of hyperparameter tuning that may be needed in actual problems. By iteratively training and testing, users can keep abreast of the model's training progress and performance on the test set. This practice provides useful experience in building and optimizing neural networks. It was also a very challenging development experience and I learned a lot.
