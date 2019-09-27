function[nn_params, J] = trainNNThroughBackProp(X, y, initial_nn_params, input_layer_size, hidden_layer_size, output_layer_size)

options = optimset('MaxIter', 50);
costFunction = @(p) nnCost(p, X, y, input_layer_size, hidden_layer_size, output_layer_size);
[nn_params, J] = fmincg(costFunction, initial_nn_params, options);

end