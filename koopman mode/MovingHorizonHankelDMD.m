%% Moving Horizon Hankel-DMD Forecasting
%Highway Traffic Dynamics: Data-Driven Analysis and Forecast 
%Allan M. Avila & Dr. Igor Mezic 2019
%University of California Santa Barbara
function [Prediction,Ptrue,MAE,MRE,RMSE,SAE,TMAE,SMAE,AvgTMAE,AvgSMAE]=...
          MovingHorizonHankelDMD(data,max_f)
%% Load Data
clc; close all;
tic
disp('Loading Data Set...')
if strcmp(data,'Poi_Day_mean')
Data=dlmread('day2.txt'); 
dtype='Mean'; hwy='Day'; delt=5;
% disp('Loading Data Set...')
% if strcmp(data,'PeMs_I10_Week_Velocity')
% Data=dlmread('I10_East_Week_Velocity_Data.txt'); 
% dtype='Velocity'; hwy='I10E Highway'; delt=5;

elseif strcmp(data,'PeMs_I5_Month_Velocity')
Data=dlmread('I5_North_Month_Velocity_Data.txt');
dtype='Velocity'; hwy='I5N Highway'; delt=5;

elseif strcmp(data,'PeMs_US101_Rain_Feb17_Velocity_Data')
Data=dlmread('US101_Rain_Feb17_Velocity_Data.txt');
dtype='Velocity'; hwy='US 101N Highway Rain'; delt=5;

elseif strcmp(data,'PeMs_SoCal_Netwk_Dec2016_Velocity_Data')
Data=dlmread('SoCal_Netwk_Dec2016_Velocity_Data.txt');
dtype='Velocity'; hwy='SoCal Network Holiday'; delt=5;

% elseif strcmp(data,'PeMs_LA_Multilane_Network_Density')
% Data=dlmread('LA_Multilane_Netwk_Dec2018_Density_Data.txt');
% dtype='Density'; hwy='LA Multi-lane Network'; delt=5;

elseif strcmp(data,'PeMs_LA_Multilane_Network_Density')  %3-5春季，6-8夏季，9-11秋季，12-2冬季
    % Data=dlmread('LA_Multilane_Netwk_Dec2018_Density_Data.txt');
    Data1 = dlmread(strcat('month\',num2str(2019),'-01.txt')); %路1=加载(XXXX)
    Data2 = dlmread(strcat('month\',num2str(2019),'-02.txt')); 
    %strcat横向连接字符串，'month\'里面所有数据连接在一起,调用时候只要用一个路径，num2str(month)转化成字符串
    Data3 = dlmread(strcat('month\',num2str(2019),'-03.txt'));
    Data4 = dlmread(strcat('month\',num2str(2019),'-04.txt'));
    Data5 = dlmread(strcat('month\',num2str(2019),'-05.txt'));
    Data6 = dlmread(strcat('month\',num2str(2019),'-06.txt'));
    Data7 = dlmread(strcat('month\',num2str(2019),'-07.txt'));
    Data8 = dlmread(strcat('month\',num2str(2019),'-08.txt'));
    Data9 = dlmread(strcat('month\',num2str(2019),'-09.txt'));
    Data10 = dlmread(strcat('month\',num2str(2019),'-10.txt'));
    Data11 = dlmread(strcat('month\',num2str(2019),'-11.txt'));
    Data12 = dlmread(strcat('month\',num2str(2019),'-12.txt'));
    Data13 = dlmread(strcat('month\',num2str(2020),'-01.txt'));
    Data14 = dlmread(strcat('month\',num2str(2020),'-02.txt')); 
    Data15 = dlmread(strcat('month\',num2str(2020),'-03.txt'));
    Data16 = dlmread(strcat('month\',num2str(2020),'-04.txt'));
    Data17 = dlmread(strcat('month\',num2str(2020),'-05.txt'));
    Data18 = dlmread(strcat('month\',num2str(2020),'-06.txt'));
    Data19 = dlmread(strcat('month\',num2str(2020),'-07.txt'));
    Data20 = dlmread(strcat('month\',num2str(2020),'-08.txt'));
    Data21 = dlmread(strcat('month\',num2str(2020),'-09.txt'));
    Data22 = dlmread(strcat('month\',num2str(2020),'-10.txt'));
    Data23 = dlmread(strcat('month\',num2str(2020),'-11.txt'));
    Data24 = dlmread(strcat('month\',num2str(2020),'-12.txt'));
    Data25 = dlmread(strcat('month\',num2str(2021),'-01.txt')); 
    Data26 = dlmread(strcat('month\',num2str(2021),'-02.txt')); 
    Data27 = dlmread(strcat('month\',num2str(2021),'-03.txt'));
    Data28 = dlmread(strcat('month\',num2str(2021),'-04.txt'));
    Data29 = dlmread(strcat('month\',num2str(2021),'-05.txt'));
    Data30 = dlmread(strcat('month\',num2str(2021),'-06.txt'));
    Data31 = dlmread(strcat('month\',num2str(2021),'-07.txt'));
    Data32 = dlmread(strcat('month\',num2str(2021),'-08.txt'));                                                                                                                                                                                             
    Data33 = dlmread(strcat('month\',num2str(2021),'-09.txt'));
    Data34 = dlmread(strcat('month\',num2str(2021),'-10.txt'));
    Data35 = dlmread(strcat('month\',num2str(2021),'-11.txt')); 
    Data36 = dlmread(strcat('month\',num2str(2021),'-12.txt'));
    Data = [Data1,Data2,Data3,Data4,Data5,Data6,Data7,Data8,Data9,Data10,Data11,Data12,Data13,Data14,Data15,Data16,Data17,Data18,Data19,Data20,Data21,Data22,Data23,Data24,Data25,Data26,Data27,Data28,Data29,Data30,Data31,Data32,Data33,Data34,Data35,Data36];
%     Data = [Data1,Data2,Data12,Data13,Data14,Data24,Data25,Data26,Data36]; dtype='hour';
%     delay=24; delt=1; hwy='spring'; hwylength=1349;xpath='x_2011.txt'; ypath='y_2011.txt'; 
dtype='Density'; hwy='Per Hour'; delt=1; xpath='x_2011.txt'; ypath='y_2011.txt'; 
end
toc

[nbx,nbt]=size(Data); % Get Data Size 1456行，366列
Delt=1; % Sampling Frequency is 5 Mins
min_s=2;% Minimum Sampling of 15 Mins
min_f=min_s;% Minimum Forecasts of 15 Mins
max_s=max_f;% Maximum Sampling=Maximum Forecasting
Prediction{max(min_f,max_f),max(min_f,max_f)}=[]; % Preallocate预先分配
MAE=zeros(max(min_f,max_f),max(min_f,max_f)); % Preallocate
MRE=zeros(max(min_f,max_f),max(min_f,max_f)); % Preallocate
RMSE=zeros(max(min_f,max_f),max(min_f,max_f)); % Preallocate
SAE{max(min_f,max_f),max(min_f,max_f)}=[]; % Preallocate
TMAE{max(min_f,max_f),max(min_f,max_f)}=[]; % Preallocate
SMAE{max(min_f,max_f),max(min_f,max_f)}=[]; % Preallocate
AvgTMAE=zeros(max(min_f,max_f),max(min_f,max_f)); % Preallocate
AvgSMAE=zeros(max(min_f,max_f),max(min_f,max_f)); % Preallocate

%% Loop over Sampling and Forecasting Windows 循环采样和预测窗口
disp('Generating Forecasts for Various Forecasting and Sampling Windows') %为各种预测和采样窗口生成预测
tic
% for f=min_f:min_f:max_f % Loop over Forecast Size Steps of 15 Mins    循环超过15分钟的预测大小
% for s=min_s:min_s:max_s % Loop over Sampling Size in Steps of 15 Mins    15分钟为步长
f=12
s=2

P=[]; PE=[]; E=[]; I=[]; R=[]; Ptrue=[]; 
Error = zeros(f,1)
rmseError =zeros(f,1)  %均方根误差
maeError = zeros(f,1)  %平均相对误差
mreError= zeros(f,1)
TMAE = zeros(f,1)
SMAE = zeros(f,1)
AvgTMAE = zeros(f,1)
AvgSMAE = zeros(f,1)
count = 0;

for t=s:f:nbt-f % Slide Window 窗口
     count = count+1;
if mod(t,300)==0 % Display Progress  显示进度
disp(['Delay=' num2str(delay) ' Forecasting Window=' num2str(f)...
    ' Sampling Window=' num2str(s) ' Current Time=' num2str(t)...
    ' Out of ' num2str(nbt-f)])
end
omega=[]; eigval=[]; Modes1=[]; bo=[]; % Clear 
Xdmd=[]; Xtrain=[]; det=[]; % Clear 

Xtrain=Data(:,t-s+1:t); % Training Data
Xfor=Data(:,t+1:t+f); % Ground Truth of Forecasted Data  预测数据的地面真实性
det=mean(Xtrain,2);% Compute and Store Time Average  计算和存储平均时间
delay=min(s-1);% Set Delays to Max Possible 将延迟设置为可能的最大值
[eigval,Modes1,bo] = H_DMD(Xtrain-repmat(det,1,size(Xtrain,2)),delay); % Compute HDMD，特征值、模式、伪逆
% [eigval,Modes1,bo] = H_DMD(Xtrain-det); % Compute HDMD
omega=log(diag(eigval))./delt; Modes1=Modes1(1:nbx,:);
% scatter(real(diag(eigval)),imag(diag(eigval))) 
% Freal=imag(omega)./(2*pi);% Compute Frequency 取虚数部分  获得频率  [703*1]
% [T,Im]=sort((1./Freal),'descend');% Sort Frequencies  T为计算出来的周期：[703*1]   
parfor time=1:s+f
Xdmd(:,time)=diag(exp(omega.*(time-1)))*bo;% Evolve
end
Xdmd=Modes1*Xdmd; % Compute Reconstructed & Forecasted Data
Xdmd=real(Xdmd+repmat(det,1,size(Xdmd,2)));
Xpre = Xdmd(:,s+1:end);  %只存储预测
% Xdmd=real(Xdmd+det);% Add the Average Back in
P=[P Xdmd(:,s+1:end)];% Only Store the Forecast
Ptrue=[Ptrue Xfor];
Perr = Xpre-Xfor;
tmp = Xpre-Xfor;
tmp2 = abs(Xpre)./abs(Xfor);
for i=1:f
    rmseError(i) = rmseError(i)+ sum(power(abs(tmp(:,i)),2));  %均方根误差，power计算A中每个元素在 B 中对应指数的幂，A和B的大小必须相同或兼容
    maeError(i) =  maeError(i)+sum(abs(tmp(:,i)));  %平均相对误差
%     mreError(i) = mreError(i)+(abs(tmp(:,i))/abs(Xfor(:,i)))
    mreError(i) = mreError(i)+sum(abs(1-tmp2(:,i)));
    Error(i) = Error(i)+ sum(tmp(:,i))/size(P,1);
end
end % Window Sliding
% (E)=tmp(:,i)
% Prediction{f,s}=P;% Store Entire Forecast for These f,w Values
E=P-Data(:,s+1:s+size(P,2));% Compute Error Matrix
% I=size(E,1)*size(E,2);% Get Total # Elements in Error matrix
% MAE(f,s)=sum(sum(abs(E)))./I;% Compute MAE
% MRE(f,s)=sum(sum(abs(E)./Data(:,s+1:s+size(P,2))))./I;% Compute MRE
% RMSE(f,s)=sqrt(sum(sum(power(abs(E),2)))./I);% Compute RMSE;
RMSE = sqrt(rmseError/(count*size(P,1))) %计算RMSE，sqrt平方根
MAE = maeError/(count*size(P,1))  %计算MAE
MRE = mreError/(count*size(P,1))  %计算MAE
% TMAE{f,s}=mean(abs(tmp(:,i)),1);% Compute TMAE,时间平均值
% SMAE{f,s}=mean(abs(tmp(:,i)),2);% Compute SMAE,平均绝对误差的空间
TMAE = mean(abs(tmp(:,i)),1);% Compute TMAE,时间平均值
SMAE = mean(abs(tmp(:,i)),2);
AvgTMAE=mean(TMAE);% Compute Avg of SMAE
AvgSMAE=mean(SMAE);% Compute Avg of SMAE
% TMAE{f,s}=mean(abs(E),1);% Compute TMAE,时间平均值
% SMAE{f,s}=mean(abs(E),2);% Compute SMAE,平均绝对误差的空间
% AvgTMAE(f,s)=mean(TMAE{f,s});% Compute Avg of SMAE
% AvgSMAE(f,s)=mean(SMAE{f,s});% Compute Avg of SMAE

for i=1:f
    Error(i) = Error(i)/(size(P,2)/f);
end
r1 = zeros(f,1);
for i=1:f
    x = P(:,i:f:end);
    y = Ptrue(:,i:f:end);
    xx = reshape(x.',size(x,1)*size(x,2),1);  %将x的行列排列成size(x,1)*size(x,2)行1列，reshape按照列取数据的
    yy = reshape(y.',size(y,1)*size(y,2),1);
    r1(i)=corr(xx,yy,'type','pearson');  %corr线性或秩相关性,X和Y中每对列的两两相关系数矩阵
end
% figure

% binscatter(xx,yy,[250,250]);%bin散点图
% colormap(gca,'jet') %颜色
% SAE{f,s}=abs(E)./mean(Data(:,s+1:s+size(P,2)),2);% Compute SAE
% TMAE{f,s}=mean(abs(E),1);% Compute TMAE
% SMAE{f,s}=mean(abs(E),2);% Compute SMAE
% AvgTMAE(f,s)=mean(TMAE{f,s});% Compute Avg of TMAE
% AvgSMAE(f,s)=mean(SMAE{f,s});% Compute Avg of SMAE

% end % End Sampling Window Loop
% end % End Forecasting WIndow Loop
title={'test'}
result_table=table(RMSE,MAE,MRE,r1);
writetable(result_table,'RMSE.csv');
disp('Forecasts Generated')
toc
tic
% disp('Generating Plots')
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%% Plot Data, Forecasts and SAE
if ~strcmp(hwy,'Day')
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,3,1)
contourf(Data,'Linestyle','none')
% xticks({}); yticks({});
h=colorbar; set(get(h,'title'),'string', {'MPH'});
title(['True Data ' hwy])

subplot(1,3,2)
contourf(Prediction{min_s,min_f},'Linestyle','none')
% xticks({}); yticks({});
h=colorbar; set(get(h,'title'),'string', {'MPH'});
title(['Forecasted Data ' hwy])

subplot(1,3,3)
contourf(SAE{min_s,min_f},'Linestyle','none') 
% xticks({}); yticks({});
h=colorbar; set(get(h,'title'),'string', {'SAE'});
title(['Scaled Absolute Error ' hwy])
elseif strcmp(hwy,'PM2.5 Day')
% d=0:minutes(5):hours(24); Time=datetime(1776,7,4)+d;
d=0:day(1):year(366); Time=datetime(2020,1,1)+d;
% Time.Format='MMMM dd,yyyy HH:mm'; Time=timeofday(Time(1:end-1));
Time.Format='MMMM dd,yyyy'; Time=timeofday(Time(1:end-1));
% PlotMultiLaneNetwork(Ptrue,Prediction{min_s,min_f},Time,min_s,min_f,hwy)

end
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%% Plot SMAE, TMAE and Histograms
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,1,1)
plot(SMAE{min_s,min_f})  %fig10,空间和时间平均绝对误差
hold on
plot(ones(1,length(SMAE{min_s,min_f})).*AvgSMAE(min_s,min_f),'m','Linewidth',2)
% title([{hwy },{['SMAE for s=' num2str(min_s) ' f=' num2str(min_f)]}])
legend('SMAE','Avg of SMAE'); xlabel('Space'); axis('tight');
% legend('SMAE','Avg of SMAE'); xticks({});xlabel('Space'); axis('tight');

subplot(3,1,2)
plot(TMAE{min_s,min_f})
hold on
plot(ones(1,length(TMAE{min_s,min_f})).*AvgTMAE(min_s,min_f),'m','Linewidth',2)
% title([{hwy },{['TMAE for s=' num2str(min_s) ' f=' num2str(min_f)]}])
legend('TMAE','Avg of TMAE'); xlabel('Time'); axis('tight');
% legend('TMAE','Avg of TMAE'); xticks({});xlabel('Time');axis('tight');

% subplot(3,1,3)
if ~strcmp(hwy,'Day')
% histogram(Data(:,min_s+1:min_s+size(Prediction{min_s,min_f},2)),...
%           'Normalization','pdf','Binwidth',.5); hold on
histogram(Prediction{min_s,min_f},'Normalization','pdf','Binwidth',.5)
% title([{hwy },{['Histogram for s=' num2str(min_s) ' f=' num2str(min_f)]}])
legend('PEMS','KMD'); axis('tight'); xlabel('MPH');
elseif strcmp(hwy,'Day')
% histogram(Data(:,min_s+1:min_s+size(Prediction{min_s,min_f},2)),...
%           'Normalization','pdf'); hold on
% histogram(Prediction{min_s,min_f},'Normalization','pdf')
% title([{hwy },{['Histogram for s=' num2str(min_s) ' f=' num2str(min_f)]}])
legend('PEMS','KMD'); axis('tight'); xlabel('Vehicle Density')
end
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%% Plot MAE,MRE,RMSE
% MAE=MAE(min_f:min_f:max_f,min_s:min_s:max_s);% Select NonZero Entries
% MRE=MRE(min_f:min_f:max_f,min_s:min_s:max_s);% Select NonZero Entries
% RMSE=RMSE(min_f:min_f:max_f,min_s:min_s:max_s);% Select NonZero Entries

if ~isscalar(MAE)
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,3,1)
% pcolor([[MAE MAE(:,end)];[MAE(end,:) MAE(end,end)]])
% xticks([1.5 length(MAE)+.5]); xticklabels([delt*min_s,delt*max_s]);
% yticks([1.5 length(MAE)+.5]); yticklabels([delt*min_s,delt*max_s]);
% xlabel('Sampling Window [Day]')
% ylabel('Forecasting Window [Day]')
% h=colorbar;
% set(get(h,'title'),'string', {'MAE'});
% title(['MAE for ' hwy])

% subplot(1,3,2)
% pcolor([[MRE MRE(:,end)];[MRE(end,:) MRE(end,end)]])
% xticks([1.5 length(MAE)+.5]); xticklabels([delt*min_s,delt*max_s]);
% yticks([1.5 length(MAE)+.5]); yticklabels([delt*min_s,delt*max_s]);
% xlabel('Sampling Window [Min]')
% ylabel('Forecasting Window [Min]')
% h=colorbar;
% set(get(h,'title'),'string', {'MRE'});
% title(['MRE for ' hwy])

subplot(1,3,3)
pcolor([[RMSE RMSE(:,end)];[RMSE(end,:) RMSE(end,end)]])
% xticks([1.5 length(MAE)+.5]); xticklabels([delt*min_s,delt*max_s]);
% yticks([1.5 length(MAE)+.5]); yticklabels([delt*min_s,delt*max_s]);
xlabel('Sampling Window [Min]')
ylabel('Forecasting Window [Min]')
h=colorbar;
set(get(h,'title'),'string', {'RMSE'});
axis('tight')
% title(['RMSE for ' hwy])
else
disp('MAE, MRE & RMSE not Matrices')
disp(['MAE=' num2str(MAE) ' MRE=' num2str(MRE) ' RMSE=' num2str(RMSE)])
end
end % End Function