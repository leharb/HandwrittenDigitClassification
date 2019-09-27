# HandwrittenDigitClassification

# Program to make predictions on the MNIST Handwritten dataset by training a neural network classifier.

# Neural network characteristics used in the program

The neural network has the following characteristics :

1. 784 input features corresponding to 1 pixel each of 28x28 image.
2. 20 units in the hidden layer.
3. 10 classes in the output layer.

For ease of classification, the digit 0 has been mapped to 10.

# Training the neural network

In octave, run the mnist.m program. It runs for 50 iterations and takes ~45 mins. The cost function is minimised using the fmincg.m program taken from Andrew Ng's Machine Learning Course on Coursera.

The trained neural network parameters have also been saved in Theta1Computed.mat (Theta paramaters for the hidden layer) and Theta2Computed.mat (Theta parameters for the output layer).
