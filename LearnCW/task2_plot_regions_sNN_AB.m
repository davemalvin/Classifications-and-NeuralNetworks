%
% Versin 0.9  (HS 06/03/2020)
%
% template script for task2_plot_regions_sNN_AB

% The greater the number of coordinates, the less 'fuzzier' the decision
% boundary will look.
noOfCoordinates = 1000;
Xplot = linspace(-2, 9, noOfCoordinates);
Yplot = linspace(-2, 7, noOfCoordinates);
% Obtain the grid vectors for the two dimensions
[Xv,Yv] = meshgrid(Xplot,Yplot);
gridX = [Xv(:),Yv(:)]; % Concatenate to get a 2-D point.

% Classify gridX using our neural network
Y = task2_sNN_AB(gridX);

figure;
% This function will draw the decision boundaries
[CC,h] =  contourf(Xplot, Yplot, reshape(Y, length(Xplot), length(Yplot)));
set(h,'LineColor','none');
title('Decision Regions sNN AB');
xlabel('x_{1}');
ylabel('x_{2}');

% Add legend
h1 = patch([0,0,0,0],[0,0,0,0], [0.35,0,1]);
h2 = patch([0,0,0,0],[0,0,0,0], [1,1,0]);
legend([h1,h2],{'Class 0','Class 1'});

% --------------------The code below is for task 2.10---------------------

% hold on
% 
% x = [1.41654, -0.983349, 2.85378, 7.58383, 1.41654];
% y = [-0.941638, 5.92176, 3.1117, 2.11629, -0.941638];
% plot(x,y,'LineWidth',1.25, 'Color',[1 0 0]);
% legend('off');
% hold on
% 
% x = [2.43627, 3.04028, 2.50204, 2.22797, 2.43627];
% y = [0.954496, 1.22806, 1.94681, 1.80233, 0.954496];
% plot(x,y,'LineWidth',1.25);
% legend('off');
% 
% legend([h1,h2],{'Class 0','Class 1'});
