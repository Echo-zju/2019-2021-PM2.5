%% Generate Koopman Traffic Modes
%Highway Traffic Dynamics: Data-Driven Analysis and Forecast 
%Allan M. Avila & Dr. Igor Mezic 2019
%University of California Santa Barbara
function [Psi]=GenerateKoopmanModes(data,mode1,mode2,save)
%% Load Data
clc; close all;
disp('Loading Data Set...')
tic
if strcmp(data,'Poi_Day_mean')
Data=dlmread('s1.txt');
delay=2; dtype='Mean'; delt=1; delx=1;hwy='day'; hwylength=34; 
toc
%% Compute KMD and Sort Modes
disp('Computing KMD via Hankel-DMD...')
tic
Avg=mean(Data,2);% Compute and Store Time Average
[eigval,Modes1,bo] = H_DMD(Data-repmat(Avg,1,size(Data,2)),delay); 
toc
disp('Sorting Modes...')
tic
%% Sampling Frequency of PeMs/NGSIM Data is 5 Minutes/Seconds.
scatter(real(diag(eigval)),imag(diag(eigval))) 
omega=log(diag(eigval))./delt;% Compute Cont. Time Eigenvalues
Freal=imag(omega)./(2*pi);% Compute Frequency
[T,Im]=sort((1./Freal),'descend');% Sort Frequencies
omega=omega(Im); Modes1=Modes1(:,Im); bo=bo(Im); % Sort Modes
toc

%% Compute and Plot Modes 
disp('Computing and Plotting Modes...')
tic
[nbx,nbt]=size(Data); % Get Data Size
time=(0:nbt-1)*delt;% Specify Time Interval
Psi=zeros(nbx,nbt,mode2-mode1+1);
res=[]
for i=mode1:mode2 % Loop Through all Modes to Plot.
psi=zeros(1,nbt);% Preallocate Time Evolution of Mode.
omeganow=omega(i);% Get Current Eigenvalue.
bnow=bo(i);% Get Current Amplitude Coefficient.
parfor t=1:length(time) 
psi(:,t)=exp(omeganow*time(t))*bnow; % Evolve for Time Length.
end
psi=Modes1(1:nbx,i)*psi;
Psi(:,:,i)=psi;
%% Plot NGSIM Modes
FONTSIZE = 35;
TICKSIZE = 28;

if strcmp(hwy,'day') % Plot NGSIM Modes
[X,Y]=meshgrid(time./1,linspace(0,hwylength,nbx));% Compute Mesh.
h=figure
warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
jFrame = get(h,'JavaFrame');	
pause(0.3);					
set(jFrame,'Maximized',1);	
pause(0.5);					
warning('on','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');	

s1=surfc(X,Y,real(psi));% Generate Surface Plot
set(s1,'LineStyle','none')% No Lines

set(gca,'position',[0.1,0.15,0.60,0.78],'TickLabelInterpreter','latex','linewidth',2.5,'FontSize',30)
title(strcat('Growing Mode #',num2str(i)),... % 'fontsize',FONTSIZE,...
                     'Interpreter','Latex','FontSize',30)
xlabel('Time (day)','Interpreter','tex','FontSize',30); 
h=colorbar;
ylabel('Monitoring station','position',[-300 -1200],'FontSize',30);%day
 
if strcmp(dtype,'Mean1')
set(get(h,'title'),'string',{'¦Ìg/m^{3} per day'},'FontSize',30);
elseif strcmp(dtype,'hour')
set(get(h,'title'),'string', {'¦Ìg/m^{3} per hour'});
end

%% Plot PeMs Modes
if strcmp(hwy,'2019day') || strcmp(hwy,'2020day')|| strcmp(hwy,'2021day')% Plot PeMs Modes
[X,Y]=meshgrid(time./1,linspace(0,hwylength,nbx));% Compute Mesh.
figure
s1=surfc(X,Y,real(psi));% Generate Surface Plot
set(s1,'LineStyle','none')% No lines
set(gca,'TickLabelInterpreter','latex','fontsize',TICKSIZE)
title({['Koopman Mode #' num2str(i)  ],[ 'Period=' num2str(T(i),'%.2f')...
    ' Hour    Growth/Decay Rate=' num2str(abs(exp(omega(i))),'%.4f')]})
title([{'Autumn Per Hour Analysis'},...
    {['Mode #' num2str(i)]}]) 
xlabel('Time (hour)','Interpreter','latex'); 
ylabel('Monitoring station (¦Ìg/m^{3})','rotation',-13,'position',[-1800 -700]);
h=colorbar; set(get(h,'title'),'string', {'¦Ìg/m^{3} per hour'})
end
end
end 
toc
disp('All Done')
end 
