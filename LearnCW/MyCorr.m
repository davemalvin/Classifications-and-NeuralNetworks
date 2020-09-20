function [corr] = MyCorr(covar)
  [row,col] = size(covar);
  corr = zeros(row,col);
  for i = 1 : row
      for j = 1 : col
          corr(i,j) = covar(i,j) / sqrt(covar(i,i) * covar(j,j));
      end
  end
end

