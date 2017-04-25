function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_new = [0;theta(2:size(theta),:)]; % Excluding the first 1 valued bias column
z = X* theta;
h = sigmoid(z);
J = (-y)' * log(h)-(1-y)'*log(1-h);
% J = (1/m) * J;

J = ((1/m).*sum(J))+(lambda/(2*m)).*sum(theta_new.^2);

%for i = 2:n
 %   grad(i) = (1/m)*(X'*(h-y));
%grad(j)=(1/m).*(sum(X'(j,:)*h-X'(j,:)*y)+lambda.*theta(j,1));

%Calculating gradient descent

grad = (X'*(h-y)+lambda*theta_new)/m;


% =============================================================

end
