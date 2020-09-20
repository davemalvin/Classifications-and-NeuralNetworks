%
% Versin 0.9  (HS 06/03/2020)
%
function task1_3(Cov)
% Input:
%  Cov : D-by-D covariance matrix (double)
% Variales to save:
%  EVecs : D-by-D matrix of column vectors of eigen vectors (double)  
%  EVals : D-by-1 vector of eigen values (double)  
%  Cumvar : D-by-1 vector of cumulative variance (double)  
%  MinDims : 4-by-1 vector (int32)  
  
  % Find the eigenvectors and eigenvalues
  [EVecs, EVals] = eig(Cov);
  
  % Extract diagonal of matrix as vector
  EVals = diag(EVals);
  
  % Sort the variances in decreasing order
  [tmp, ridx] = sort(EVals, 1, 'descend');
  
  % Store sorted eigvals in EVals, and its corresponding eigvects in EVecs
  EVals = tmp;
  EVecs = EVecs(:,ridx);
  
  % First element of each eigvect should be non-negative
  numOfEVals = size(EVals,1);
  for i = 1:numOfEVals
      if (EVecs(1,i) < 0)
          EVecs(1,i) = EVecs(1,i) * -1;
      end
  end
  
  % Cumvar is the cumulative sum of variances
  Cumvar = cumsum(EVals);
  
  % Last element of Cumvar is the total variance
  CVratio = Cumvar / Cumvar(numOfEVals);
  percentList = {0.7, 0.8, 0.9, 0.95};
  MinDims = zeros(4,1);
  j = 1;
  for i = 1:numOfEVals
      if (CVratio(i) >= percentList{j})
          MinDims(j) = i;
          % If the no. of PCA dimension to cover 95% of total variance 
          % has been found, exit the for loop.
          if (MinDims(4,1) > 0)
            break
          end
          j = j + 1;
      end
  end

  save('t1_EVecs.mat', 'EVecs');
  save('t1_EVals.mat', 'EVals');
  save('t1_Cumvar.mat', 'Cumvar');
  save('t1_MinDims.mat', 'MinDims');
end
