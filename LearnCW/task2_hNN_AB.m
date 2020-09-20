%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_hNN_AB(X)
% Input:
%  X : N-by-D matrix of input vectors (in row-wise) (double)
% Output:
%  Y : N-by-1 vector of output (double)
    
    % Get the weights for Polygon_A
    file = importdata('task2_hNN_A_weights.txt',' ');
    Wa = file.data; % Weights as a column vector
    W1a = Wa(1:3);
    W2a = Wa(4:6);
    W3a = Wa(7:9);
    W4a = Wa(10:12);
    
    % Weights where the outside of Polygon_A is class 1
    W1a_x = W1a * -1;
    W2a_x = W2a * -1;
    W3a_x = W3a * -1;
    W4a_x = W4a * -1;
    
    % Get the weights for Polygon_B
    W1b = [-3.109498449; 2.859881436; 1];
    W2b = [5.201620101; -0.7323339924; -1];
    W3b = [3.712260491; -0.2104438642; -1];
    W4b = [1.644001166; -0.4958300972; 1];
    
    W2b_x = W2b * -1;
    
    % Weights for perceptron acting as AND3, AND4 and OR5 gate
    And3 = [-2.5,1,1,1]'; 
    And4 = [-3.5,1,1,1,1]';
    Or5 = [0,1,1,1,1,1]'; 
    
    % Get the size of X; We assume D=2 hereafter.
    [N,~] = size(X);
    
    % Initialize output vector
    Y = zeros(N,1);
    
    % Implement neural network
    for i = 1:N
        p1a = task2_hNeuron(W1a,X(i,:));
        p2a = task2_hNeuron(W2a,X(i,:));
        p4a = task2_hNeuron(W4a,X(i,:));
    
        p1a_x = task2_hNeuron(W1a_x,X(i,:));
        p2a_x = task2_hNeuron(W2a_x,X(i,:));
        p3a_x = task2_hNeuron(W3a_x,X(i,:));
        p4a_x = task2_hNeuron(W4a_x,X(i,:));
        
        p1b = task2_hNeuron(W1b,X(i,:));
        p2b = task2_hNeuron(W2b,X(i,:));
        p3b = task2_hNeuron(W3b,X(i,:));
        p4b = task2_hNeuron(W4b,X(i,:));
        
        p2b_x = task2_hNeuron(W2b_x,X(i,:));
        
        P = [p1b,p2b,p1a,p4a_x];
        z1 = task2_hNeuron(And4,P);
        
        P = [p4a,p2b,p1a,p2a_x];
        z2 = task2_hNeuron(And4,P);
        
        P = [p1b,p4b,p3b,p1a_x];
        z3 = task2_hNeuron(And4,P);
        
        P = [p3b,p2b_x,p1a];
        z4 = task2_hNeuron(And3,P);
        
        P = [p2a,p4a,p3a_x];
        z5 = task2_hNeuron(And3,P);
        
        Z = [z1,z2,z3,z4,z5];
        Y(i,:) = task2_hNeuron(Or5,Z);
    end
end
