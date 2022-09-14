%PM2.5 Dynamics: PM2.5 Analysis
%Zhejiang University

%% Instructions for Koopman Modes.
% To generate Koopman modes call on the function:
% Modes=GenerateKoopmanModes(Data,Mode1,Mode2,Save)

%Inputs Required:
% Data is a string containing the name of the data set.

% Mode1 and Mode2 are integers indicating which modes to produce.
% Ordered by their period of oscilaliton from slowest to fastest.
% Mode1 can be < or = Mode2.

% Save is a logical (0 or 1) indicating the modes to be saved as jpeg's. 


% The Following are correctly named data sets for generating Koopman Modes:
% 1. 2019day
% 2. 2020day
% 3. 2021day
% 4. s1
% 5. s2
% 6. s3
% 7. s4

%Outputs Returned:
% Modes is an n by m by #modes sized  array. 
% For example Modes(:,:,i) contains the i'th mode.

%Plots Generated:
% The funciton will generate plots of the desired Koopman Modes.

% Examples:
clc; clear variables; close all;
Poi_Daily=GenerateKoopmanModes('Poi_Day_mean',1,1,0);
% 
% clc; clear variables; close all;
