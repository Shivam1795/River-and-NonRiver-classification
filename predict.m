 function pr = predict(theta, eg2)

% This function is used to predict the house price for a particular example
% via the theta's obtained from gradient descent

% First need to normalize the values of the features
eg1 = [eg2(:,:,1) eg2(:,:,2) eg2(:,:,3) eg2(:,:,4)];
eg1 = double(eg1);

% eg1 = ((eg1-mu)./stddev);

eg1 = [ones(size(eg1,1),1) eg1];
pr = sigmoid(eg1*theta);
end