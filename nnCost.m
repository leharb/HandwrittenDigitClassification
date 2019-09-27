function[J, grad] = nnCost(nn_params, X, y, input_layer_size, hidden_layer_size, output_layer_size)

m = size(X,1);
X = [ones(m,1) X];

Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size+1)), hidden_layer_size, input_layer_size+1);
Theta2 = reshape(nn_params(hidden_layer_size*(input_layer_size+1)+1:end), output_layer_size, hidden_layer_size+1);

%cost
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = X; %m x (n+1)
z2 = a1*Theta1'; %m x hidden
a2 = sigmoid(z2); %m x hidden
a2 = [ones(m,1) a2]; %m x (hidden+1)
z3 = a2*Theta2'; %m x output_layer_size
h = a3 = sigmoid(z3);  %m x output_layer_size

Y = zeros(m, output_layer_size);

for eg = 1:m
	Y(eg, y(eg))=1;
end

J = (1/m) * sum(sum(-Y.*log(h) - (1-Y).*log(1-h)))

delta1 = zeros(hidden_layer_size, input_layer_size + 1);
delta2 = zeros(output_layer_size, hidden_layer_size + 1);

for eg = 1:m
	%step 1 : feedforward pass
	a1 = X(eg,:); %1x(n+1)
	z2 = a1*Theta1'; %1xhidden_layer_size
	a2 = sigmoid(z2);%1xhidden_layer_size
	a2 = [1 a2];% 1x(hidden_layer_size+1)
	z3 = a2 * Theta2'; % 1xoutput_layer_size
	a3 = sigmoid(z3);% 1xoutput_layer_size
	
	%step 2
	smallDelta3 = a3 - Y(eg,:);%1xoutput_layer_size
	smallDelta3 = smallDelta3(:); %output_layer_size x 1
	
	%step 3
	firstProduct = Theta2' * smallDelta3; %hidden_layer_size+1 x 1
	smallDelta2 = firstProduct(2:end).*(sigmoidGradient(z2(:))); %hidden_layer_size x 1
	
	%step 4
	delta1 = delta1 + smallDelta2*a1;%hidden_layer_size x (input_layer_size+1)
	delta2 = delta2 + smallDelta3*a2;%output_layer_size x (hidden_layer_size+1)				
end
	
Theta1_grad = delta1/m;
Theta2_grad = delta2/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end