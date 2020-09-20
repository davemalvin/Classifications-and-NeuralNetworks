%
% Versin 0.9  (HS 06/03/2020)
%
function task1_mgc_cv(X, Y, CovKind, epsilon, Kfolds)
% Input:
%  X : N-by-D matrix of feature vectors (double)
%  Y : N-by-1 label vector (int32)
%  CovKind : scalar (int32)
%  epsilon : scalar (double)
%  Kfolds  : scalar (int32)
%
% Variables to save
%  PMap   : N-by-1 vector of partition numbers (int32)
%  Ms     : C-by-D matrix of mean vectors (double)
%  Covs   : C-by-D-by-D array of covariance matrices (double)
%  CM     : C-by-C confusion matrix (double)

  noOfClasses = max(Y);
  dim_X = size(X,2);
  noOfSamples = size(X,1); 
  
  % Find the number of samples for each class
  Nc = zeros(1,noOfClasses);
  for i = 1:noOfSamples
      class = Y(i);
      Nc(class) = Nc(class) + 1;
  end
  
  % Find the number of samples of class c assigned to first partition
  Mc = zeros(1,noOfClasses);
  for c = 1:noOfClasses
      Mc(c) = floor(double(Nc(c)) / double(Kfolds));
  end
  
  % Find the number of remaining samples to be added to last partition
  last = zeros(1,noOfClasses);
  for c = 1:noOfClasses
      last(c) = Nc(c) - Mc(c) * (Kfolds - 1);
  end
  
  % For each i, assign the first Mc(i) samples of class j to Partition 1,
  % and assign next Mc(i) samples to Partition 2 and so on...
  PMap = zeros(noOfSamples,1);
  PNum = ones(1,noOfClasses); % To know current partition number for each class
  temp = Mc;
  for i = 1:noOfSamples
      j = Y(i);
      % Move to next partition if number of samples to be inserted to
      % current partition reaches 0
      if (temp(j) == 0)
          temp(j) = Mc(j);
          PNum(j) = PNum(j) + 1;
          % If partition is last partition, allocate remaining samples
          if (PNum(j) == Kfolds)
              temp(j) = last(j);
          end
      end
      PMap(i) = PNum(j);
      temp(j) = temp(j) - 1;
  end
  
  % For each fold p
  for p = 1:Kfolds
      % Split samples to those that belong to p (TEST) and those that don't
      % belong to p (TRAINING); Store it in Xtest and Xtrain respectively.
      nTrain = noOfSamples - sum(Mc);
      if (p == Kfolds)
          nTrain = noOfSamples - sum(last);
      end
      nTest = noOfSamples - nTrain;
      
      Xtrain = zeros(nTrain, dim_X);
      Xtest = zeros(nTest, dim_X);
      % The corresponding labels/species is stored in Y_notIn_p and Y_In_p
      Ytrain = zeros(nTrain, 1);
      Ytest = zeros(nTest, 1);
      % Initialise index for storing
      ctrNotIn = 1;
      ctrIn = 1;
      for i = 1:noOfSamples
          if (PMap(i) ~= p)
              Xtrain(ctrNotIn,:) = X(i,:);
              Ytrain(ctrNotIn,1) = Y(i,1);
              ctrNotIn = ctrNotIn + 1;
          end
          if (PMap(i) == p)
              Xtest(ctrIn,:) = X(i,:);
              Ytest(ctrIn,1) = Y(i,1);
              ctrIn = ctrIn + 1;
          end
      end

      % For each class c, estimate the mean vect and cov matrix from the
      % samples that DO NOT belong to partition p
      Ms = [];
      Covs = zeros(dim_X,dim_X);
      for c = 1:noOfClasses
          % X_In_c is the training set that belongs to class c
          X_In_c = Xtrain(Ytrain == c,:);
          
          % Ms(c,:) is the sample mean vector for class c
          mu = MyMean(X_In_c);
          Ms = [Ms; mu];
     
          % Estimate cov matrix
          cov = MyCov(X_In_c);
          
          % Regularise cov matrix
          I = eye(dim_X,dim_X);
          regularised = cov + epsilon * I;
          
          % Condition for diagonal cov matrix
          if (CovKind == 2)
              regularised = regularised .* I;
          end
          % Covs(:,:,c) is the cov mtrx for class c (after regularisation)
          Covs(:,:,c) = regularised;
      end
      
      % Condition for shared cov matrix
      if (CovKind == 3)
          totalCov = zeros(dim_X,dim_X);
          for i = 1:noOfClasses
              totalCov = totalCov + Covs(:,:,i);
          end
          sharedCov = (1 / double(noOfClasses)) * totalCov;
          for i = 1:noOfClasses
              Covs(:,:,i) = sharedCov;
          end
      end

      % Initialise posterior probabilities
      pp = zeros(nTest, noOfClasses);
      % Compute the log likelihood
      for c = 1:noOfClasses
          mu_c = Ms(c,:);
          covar_c = Covs(:,:,c);
          lik_c = gaussianMV(mu_c, covar_c, Xtest);
          prior = length(find(Ytest == c)) / length(Xtest);
          pp(:,c) = log(lik_c) + log(prior);
      end
      % Ypreds is the predicted labels for Xtest 
      [~,Ypreds] = max(pp,[],2);
      
      % Compute confusion matrix
      [CM,acc] = comp_confmat(Ytest, Ypreds, noOfClasses);
      fn_CM = strcat('t1_mgc_',int2str(Kfolds),'cv',int2str(p),'_ck',int2str(CovKind),'_CM.mat');
      save(fn_CM, 'CM');
      
      % Calculate final confusion matrix where each elem is a relative freq
      noOfObservations = sum(CM,'all');
      CM = CM / noOfObservations;
      L = Kfolds + 1;

      % Reshape cov matrix to C-by-D-by-D from D-by-D-by-C
      Covs = permute(Covs, [3 2 1]);
      
      fn_FCM = strcat('t1_mgc_',int2str(Kfolds),'cv',int2str(L),'_ck',int2str(CovKind),'_CM.mat');
      fn_Ms = strcat('t1_mgc_',int2str(Kfolds),'cv',int2str(p),'_Ms.mat');
      fn_Covs = strcat('t1_mgc_',int2str(Kfolds),'cv',int2str(p),'_ck',int2str(CovKind),'_Covs.mat');
      
      save(fn_Ms, 'Ms');
      save(fn_Covs, 'Covs');
      save(fn_FCM, 'CM');
      
  end

  fn_PMap = strcat('t1_mgc_',int2str(Kfolds),'cv_PMap.mat');
  save(fn_PMap, 'PMap');
end
