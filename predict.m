function[predictions] = predict(X, Theta1, Theta2)

m = size(X, 1);
n = size(X, 2);

predictions = zeros(m,1);

%start feedforward propagation

a1 = [ones(m,1) X]; %m x (n+1)
z2 = a1 * Theta1'; %m x hidden
a2 = sigmoid(z2); % m x hidden
a2 = [ones(m,1) a2];%m x (hidden+1)

output = sigmoid(a2 * Theta2'); % m x k

for eg = 1:m
	[val ival] = max(output(eg,:));
	predictions(eg) = ival;
end

end