function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
m = length(y);

J = sum(((X * theta)-y).^2);
J = (1/(2*m)) * J ;
% Squared Errors = (h - y).^2  where h = X* theta
% Adding regularization to the Linear Regression

thetaNonZero = [0; theta(2:end)];
J = J + (lambda/(2*m)) * sum(thetaNonZero.^2);

% Gradient Calculation

for j = 1:length(grad)
    grad(j) = (1/m)*sum(((X * theta)-y).* X(:,j)) + (lambda/m)* thetaNonZero(j) ;
end

%G = (lambda/m) .* theta';
%G(1) = 0; % this is always 0

%grad = ((1/m) .* X' * (X*theta - y)) + G;
% =========================================================================

grad = grad(:);

end
