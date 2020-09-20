function [covar] = MyCov(X)
  % N is number of samples
  [N,D] = size(X);
  
  % Subtract off the mean for each dim. so each dim. will have zero mean
  % Calculate the mean for each column
  X_mean = MyMean(X); 
  
  % Mean shift the original matrix 
  X_shift = bsxfun(@minus, X, X_mean);
  
  % Compute covariance matrix
  covar = 1 / (N) * (X_shift' * X_shift);
end

