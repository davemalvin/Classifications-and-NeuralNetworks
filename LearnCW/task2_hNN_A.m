%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_hNN_A(X)
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
% Output:
%  Y : N-by-1 vector of output (double)

    file = importdata('task2_hNN_A_weights.txt',' ');
    W = file.data; % Weights as a column vector
    
    % Get the size of X; We assume D=2 hereafter.
    [N,~] = size(X);
    
    % Initialize output vector
    Y = zeros(N,1);
    for i = 1:N
        z1 = task2_hNeuron(W(1:3),X(i,:));
        z2 = task2_hNeuron(W(4:6),X(i,:));
        z3 = task2_hNeuron(W(7:9),X(i,:));
        z4 = task2_hNeuron(W(10:12),X(i,:));
        
        Z = [z1,z2,z3,z4];
        Y(i,:) = task2_hNeuron(W(13:17),Z);
    end
end
