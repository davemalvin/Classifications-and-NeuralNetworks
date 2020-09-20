%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_sNeuron(W, X)
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
%  W : (D+1)-by-1 vector of weights (double)
% Output:
%  Y : N-by-1 vector of output (double)

    % Initialise output vector Y
    [N,D] = size(X);
    Y = zeros(N,1);
    
    % Convert X to an augmented one
    Xa = zeros(N,D+1);
    Xa(:,1) = 1;
    Xa(:,2:(D+1)) = X;
    
    a = zeros(N,1);
    for i = 1:N
        x = Xa(i,:);
        a(i) = x * W;
        % Implement logistic sigmoid function
        Y(i) = 1 / (1 + exp(-a(i)));
    end

end
