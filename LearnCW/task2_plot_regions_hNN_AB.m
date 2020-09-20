%
% Versin 0.9  (HS 06/03/2020)
%
% template script for task2_plot_regions_hNN_AB.m

% The greater the number of coordinates, the less 'fuzzier' the decision
% boundary will look.
noOfCoordinates = 1000;
Xplot = linspace(-2, 9, noOfCoordinates);
Yplot = linspace(-2, 7, noOfCoordinates);
% Obtain the grid vectors for the two dimensions
[Xv,Yv] = meshgrid(Xplot,Yplot);
gridX = [Xv(:),Yv(:)]; % Concatenate to get a 2-D point.

% Classify gridX using our neural network
Y = task2_hNN_AB(gridX);

figure;
% This function will draw the decision boundaries
[CC,h] =  contourf(Xplot, Yplot, reshape(Y, length(Xplot), length(Yplot)));
set(h,'LineColor','none');
title('Decision Regions hNN AB');
xlabel('x_{1}');
ylabel('x_{2}');

% Add legend
h1 = patch([0,0,0,0],[0,0,0,0], [0.35,0,1]);
h2 = patch([0,0,0,0],[0,0,0,0], [1,1,0]);
legend([h1,h2],{'Class 0','Class 1'});
        



