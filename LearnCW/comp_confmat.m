function [CM, acc] = comp_confmat(Ytrues, Ypreds, C)
% Input:
%   Ytrues : N-by-1 ground truth label vector
%   Ypreds : N-by-1 predicted label vector
% Output:
%   CM : C-by-C confusion matrix, where CM(i,j) is the number of samples whose target is the ith class that was classified as j
%   acc : accuracy (i.e. correct classification rate)

    % Initalise CM
    CM = zeros(C,C);
   
    [N, ~] = size(Ytrues);
    
    % populate the confusion matrix by iterating through K 
    for c = 1:C
       preds = Ypreds(Ytrues == c);
       preds = preds';
       for p = preds
           tmp = CM(c, p) + 1;
           CM(c, p) = tmp;
       end
    end
    
    % get the accuracy
    acc = trace(CM) / N;
   
end


