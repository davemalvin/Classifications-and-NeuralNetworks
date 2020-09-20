%
% Versin 0.9  (HS 06/03/2020)
%
function [Y] = task2_sNN_AB(X)
% Input:
%  X : N-by-D matrix of input vectors (double)
% Output:
%  Y : N-by-1 vector of output (double)
    
    scaleUp = 1000;
    % Get the weights for Polygon_A and scale it up by factor of scaleUp
    file = importdata('task2_hNN_A_weights.txt',' ');
    Wa = file.data; % Weights as a column vector
    W1a = Wa(1:3) * scaleUp;
    W2a = Wa(4:6) * scaleUp;
    W3a = Wa(7:9) * scaleUp;
    W4a = Wa(10:12) * scaleUp;
    
    % Weights where the outside of Polygon_A is class 1
    W1a_x = W1a * -1;
    W2a_x = W2a * -1;
    W3a_x = W3a * -1;
    W4a_x = W4a * -1;
    
    % Get the weights for Polygon_B and scale it up by a factor of scaleUp
    W1b = [-3.109498449; 2.859881436; 1] * scaleUp;
    W2b = [5.201620101; -0.7323339924; -1] * scaleUp;
    W3b = [3.712260491; -0.2104438642; -1] * scaleUp;
    W4b = [1.644001166; -0.4958300972; 1] * scaleUp;
    
    W2b_x = W2b * -1;
    
    % Weights for perceptron acting as AND3, AND4 and OR5 gate scaled-up
    And3 = [-2.5,1,1,1]' * 200; 
    And4 = [-3.5,1,1,1,1]' * 200;
    Or5 = [0,1,1,1,1,1]' * 200; 
    
    % Get the size of X; We assume D=2 hereafter.
    [N,~] = size(X);
    
    % Initialize output vector
    Y = zeros(N,1);
    
    % Implement neural network
    for i = 1:N
        p1a = task2_sNeuron(W1a,X(i,:));
        p2a = task2_sNeuron(W2a,X(i,:));
        p4a = task2_sNeuron(W4a,X(i,:));
    
        p1a_x = task2_sNeuron(W1a_x,X(i,:));
        p2a_x = task2_sNeuron(W2a_x,X(i,:));
        p3a_x = task2_sNeuron(W3a_x,X(i,:));
        p4a_x = task2_sNeuron(W4a_x,X(i,:));
        
        p1b = task2_sNeuron(W1b,X(i,:));
        p2b = task2_sNeuron(W2b,X(i,:));
        p3b = task2_sNeuron(W3b,X(i,:));
        p4b = task2_sNeuron(W4b,X(i,:));
        
        p2b_x = task2_sNeuron(W2b_x,X(i,:));
        
        P = [p1b,p2b,p1a,p4a_x];
        z1 = task2_sNeuron(And4,P);
        
        P = [p4a,p2b,p1a,p2a_x];
        z2 = task2_sNeuron(And4,P);
        
        P = [p1b,p4b,p3b,p1a_x];
        z3 = task2_sNeuron(And4,P);
        
        P = [p3b,p2b_x,p1a];
        z4 = task2_sNeuron(And3,P);
        
        P = [p2a,p4a,p3a_x];
        z5 = task2_sNeuron(And3,P);
        
        Z = [z1,z2,z3,z4,z5];
        output = task2_sNeuron(Or5,Z);
        
        % Add threshold to final output
        if output >= 0.6
            Y(i,:) = 1;
        else
            Y(i,:) = 0;
        end
    end
end
