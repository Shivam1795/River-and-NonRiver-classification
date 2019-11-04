function CostVal = cost(X,y,theta)

m = length(y); % number of training examples

CostVal = -(1/m)*sum(y'*log(sigmoid(X*theta)) + (1-y)'*log(1-sigmoid(X*theta)));
end