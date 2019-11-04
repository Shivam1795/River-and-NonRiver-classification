clear all;
clc;
close all;

river = csvread('river_dataset\river_coordinates.csv');

[row, column] = size(river);

I1 = imread('images\1.gif');
I2 = imread('images\2.gif');
I3 = imread('images\3.gif');
I4 = imread('images\4.gif');
[row, column] = size(river);
for itr = 1:row
    River(itr, 1) = I1(river(itr)); 
    River(itr, 2) = I2(river(itr));
    River(itr, 3) = I3(river(itr));
    River(itr, 4) = I4(river(itr));
    River(itr, 5) = 1;
end
river = csvread('river_dataset\non-river_coordinates.csv');
for itr = row+1:row+100
    River(itr, 1) = I1(river(itr)); 
    River(itr, 2) = I2(river(itr));
    River(itr, 3) = I3(river(itr));
    River(itr, 4) = I4(river(itr));
    River(itr, 5) = 0;
end
csvwrite('river_dataset\river_data.csv',River);

%% Load Data and Initialize Variables
fprintf('Loading data and initializing variables');
X = River(:, 1:4); % FeaturesX(1)=R,X(2)=G,X(3)=B,X(4)=I
X = double(X);
y = River(:, 5);
y = double(y);
m = length(y); % Number of training examples
d = size(X,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
theta(:,1) = 10;
alpha = 0.03; % Learning rate
numIters = 1000; % How long gradient descent should run for
fprintf('...done\n');

%% Feature Normalization
% fprintf('Normalizing Features for gradient descent');
% [X, mu, stddev] = featureNormalize(X);
% fprintf('...done\n');

%% Compute the Cost Function
fprintf('Calculating theta via gradient descent');
X = [ones(m,1) X]; % Add a col of 1's for the x0 terms

[theta, CostHistory] = gradientDescent(X, theta, y, alpha, numIters);
fprintf('...done\n');

%% Result
test_data = load('river_dataset\testdata.mat','River');
output = zeros(512,512);
River = test_data.River;
for i = 1:512
    for j = 1:512
        x = River(i,j,1:4);
        if (predict(theta, x)>= 0.5)
            output(i,j) = 255;
        end
    end
end
 imwrite(output,'output_images\out_image_without_normal.png');           
