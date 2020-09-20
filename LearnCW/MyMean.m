  function [result] = MyMean(m)
    [row, col] = size(m);
    
    result = zeros(1, col);
    
    for i = 1:col
        colSum = sum(m(:,i));
        colMean = colSum / row;
        result(:,i) = colMean;
    end
end

