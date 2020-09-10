%  A simple machine learning example.
%
%  You've gathered data about various boxes of chocolates: how much they cost, 
%  and how many chocolates they have contained. You want to be able to predict
%  how many chocolates you'll get in a box for any given price. 

clear; close all; clc;

%  First, we'll put the data about prices (x) and number of chocolates (y) 
%  into two separate vectors. 
x = [1, 1.30, 1.45, 1.75, 2, 2, 2.30, 2.50, 2.55, 2.75, 2.80, 3, 3, 3.20, 3.35, 3.35, 3.50, 3.55, 3.60, 3.80, 4, 4, 4.20, 4.30, 4.50]
y = [3, 3, 4, 1, 3, 5, 5, 6, 4, 7, 5, 3, 7, 7, 5, 6, 8, 7, 5, 8, 10, 9, 9, 8, 9]

fprintf('\nThere is our data.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Next, plot the data to get a sense of its shape.
scatter(x, y)
xlabel('Price')
ylabel('Number of chocolates')
axis([0, 5, 0, 11])

%  Linear regression will do. Let's define our cost function.
function J = costFunction(X, y, theta)
%  X is the design matrix, y is a vector of y values, theta is a vector 
%  of theta0 and theta1.
m = size(X, 1);			% The number of rows, or training examples in our data.
predictions = X * theta;	% Our predicted y value for each x value in our data.
sqrErrors = (predictions - y) .^ 2;  %  How far off our predicted y is from each actual y.
J = 1/(2*m) * sum(sqrErrors); 	     % How far off our predictions are, typically. 
end;

%  To test different thetas, have to make X, the design matrix.
X = [ones(length(x), 1), x']

%  Must also transpose y.
y = y'

%  Initialize your first guess as to what theta0 and theta1 might be.
theta = [0; 1]

%  Calculate J for those values of theta.
costFunction(X, y, theta)

fprintf('\nThere is our design matrix, y, initial theta, and cost for theta=[0;1].\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Gradient descent will give us the values of theta that result in the lowest J.
function[theta, J_values] = gradientDescent(X, y, theta, alpha, iterations)
%  Will return the theta0 and theta1 that minimize J, and all J-values computed
%  along the way (to make sure it went down each time). 
m = length(y);
J_values = zeros(iterations, 1);
for i=1:iterations,
theta = theta - (alpha * (1/m) * ((X*theta - y)' * X)');
J_values(i) = costFunction(X, y, theta);
end;
end;

%  Try out gradient descent, with alpha = .01, at 1500 iterations. 
gradientDescent(X, y, theta, .01, 1500)
%  You should also test for the following values of alpha. 
gradientDescent(X, y, theta, .03, 1500)
gradientDescent(X, y, theta, .1, 1500)
gradientDescent(X, y, theta, .3, 1500)
gradientDescent(X, y, theta, 1, 1500)

fprintf('\nThere are our theta values for gradient descent when alpha=.01, .03, .1, .3, and 1.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  We'll stick with the one that causes J to decrease on every iteration, before it
%  converges on a single value. In this case, it's alpha=.3, which gives us 
%  theta0 = .0005 and theta1 = 1.99. Overlay this line onto the scatterplot.

%  First, calculate the predicted y values for this hypothesis. 
theta = [.0005; 1.99]
predicted_y = X * theta

fprintf('\nThere are our predictions, given our new hypothesis.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Then, plot. 
scatter(x, y)
hold on
plot(x, predicted_y, 'r')
xlabel('Price')
ylabel('Number of chocolates')

%  Looks pretty good! Our function y = .0005 + 1.99x can now be used to make predictions
%  about new data. 

fprintf('\nThere is a plot of our hypothesis.\n');


