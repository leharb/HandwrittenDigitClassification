%load data
data = load('mnist.mat');
Xtrain = double(data.trainX);
yTrain = double(data.trainY(:));
XTest = double(data.testX);
yTest = double(data.testY(:));

%convert the class 0 to class 10 for ease of computation

idx = find(yTrain==0);
yTrain(idx) = 10;

idx = find(yTest==0);
yTest(idx) = 10;

%size of input layer/no. of features
input_layer_size = size(Xtrain, 2);

%size of hidden layer
hidden_layer_size = 20;

%size of output layer/no. of classes
output_layer_size = 10;

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

initial_nn_params = [Theta1(:); Theta2(:)];

[nn_params, J] = trainNNThroughBackProp(Xtrain, yTrain, initial_nn_params, input_layer_size, hidden_layer_size, output_layer_size);

Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size+1)), hidden_layer_size, input_layer_size+1);
Theta2 = reshape(nn_params(hidden_layer_size*(input_layer_size+1)+1:end), output_layer_size, hidden_layer_size+1);

predictions = predict(XTest, Theta1, Theta2);

total = size(yTest, 1);
correct = sum(yTest==predictions);
accuracy = (correct/total)*100



