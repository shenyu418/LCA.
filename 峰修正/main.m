%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res= xlsread('50组纯福测试2.xlsx');
res=res';
X_blank = xlsread('jidi.xlsx');
Y = xlsread('outputd2');
X_blank1 =xlsread('验证');
X_blank1=X_blank1';
Y_blank1=xlsread('验证output');
Y_blank1=Y_blank1';

% %%  划分训练集和测试集
% temp = randperm(50);
% 
% P_train = res(temp(1: 35), 1 : 1023)';
% T_train = res(temp(1: 35), 1024)';
% M = size(P_train, 2);
% 
% P_test = res(temp(36: end), 1 : 1023)';
% T_test = res(temp(36: end), 1024)';
% N = size(P_test, 2);


 P_train=[];
  P_test= [];
  T_train=[];
   T_test=[];
for i= 1 : 5
    X1 = res((((i-1)*10)+1 : i*10), 1 :1023 );
    Y1 = Y((((i-1)*10)+1 : i*10),  : );
    temp = randperm(10);%1代表多少行，2代表多少列
%训练集—
    P_train1= X1(temp(1:7),:);%冒号代表取出来是整行或者整列，'代表转置
    P_test1 = X1(temp(8:end),:);
    
% 测试集—
    T_train1= Y1(temp(1:7),:);
    T_test1 = Y1(temp(8:end),:);
    
    % 将P_train1按行拼接到P_train
    P_train = vertcat(P_train, P_train1);
    P_test = vertcat(P_test, P_test1);
    T_train = vertcat(T_train, T_train1);
    T_test = vertcat(T_test, T_test1);
%    P_train =P_train'; 
%     P_test=P_test';
%     T_train=T_train';
%     T_test=T_test';
end
clear P_train1 P_test1 T_train1 T_test1 temp X1 Y1 i;
%    P_train =P_train'; 
%     P_test=P_test';
%     T_train=T_train';
%     T_test=T_test';
% M = size(P_train,2);
% N = size(P_test,2);
% P_train=P_train-jidi';
% P_test=P_test-jidi';
%%
% %% CARS降维
% 
% egg_y = [ P_train;P_test];
% egg_labels_y = [T_train;T_test];
% 
% % 提取特征波段，SelectedVariables是得到的变量的列编号
% %MCCV=plsmccv(egg_y,egg_labels_y,100,'center',50,0.9,1);
% sCARS=scarspls(egg_y,egg_labels_y,100,10,'center',10);% CARS的简化版本
% % scarspls.mat可以重现变量选择结果
% plotcars(sCARS);
% SelectedVariables=sCARS.vsel;
% % 得到CARS后的波段矩阵
% CARS_labels = SelectedVariables;
%  P_train =  P_train(:,CARS_labels);
% P_test = P_test(:,CARS_labels);


   P_train =P_train'; 
    P_test=P_test';
    T_train=T_train';
    T_test=T_test';
M = size(P_train,2);
N = size(P_test,2);



%%  创建网络
net = newff(P_train, T_train, 10);
%%  设置训练参数
net.trainParam.epochs = 1000;     % 迭代次数 1000
net.trainParam.goal = 1e-6;       % 误差阈值1e-6
net.trainParam.lr = 0.001;         % 学习率0.01
net.trainFcn = 'trainlm';

%%  训练网络 需要样本数在列
net = train(net, P_train, T_train);

%%  仿真测试 样本数在列，训练集
t_sim1 = sim(net, P_train);
t_sim2 = sim(net, P_test );

%%  均方根误差
error1 = sqrt(sum((t_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((t_sim2 - T_test ).^2) ./ N);
RMSEC1 = sqrt(sum((t_sim1 - T_train).^2) ./ (M-1));
RMSEC2 = sqrt(sum((t_sim2 - T_test).^2) ./ (N-1));
disp(['均方根误差RMSEP为：  ',num2str(error2)])
disp(['均方根误差RMSECV为：  ',num2str(error1)])

%%  绘图
figure
subplot(2, 1, 1)
plot(1: M, T_train, 'r-*', 1: M, t_sim1, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])

grid

subplot(2, 1, 2)
plot(1: N, T_test, 'r-*', 1: N, t_sim2, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid
figure;
plotregression(T_test,t_sim2,['回归图']);

%%  分割线
disp('**************************')
disp(['下列是输出', num2str(i)])
disp('**************************')

%%  相关指标计算
%相关系数
covariance_XY = sum((T_train - mean(T_train)) .* (((t_sim1)) - mean(t_sim1))) / (length(T_train)- 1);
std_X = sqrt(sum((T_train - mean(T_train)).^2) / (length(T_train) - 1));
std_Y = sqrt(sum((t_sim1 - mean(t_sim1)).^2) / (length(t_sim1) - 1));
r = covariance_XY / (std_X * std_Y  );
t_sim11=10.^(t_sim1);
T_train1=10.^(T_train);
bias1= mean(T_train - t_sim1);
bias2= mean(T_test - t_sim2);
% 决定系数 R2
R1 = 1 - sum((T_train - t_sim1).^2) / sum((T_train - mean(T_train)).^2);
R2 = 1 - sum((T_test  - t_sim2).^2 )/ sum((T_test  - mean(T_test )).^2);


disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

% 平均绝对误差 MAE
mae1 = sum(abs(t_sim1 - T_train)) ./ M ;
mae2 = sum(abs(t_sim2 - T_test)) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

% 平均相对误差 MBE
mbe1= sum(t_sim1 - T_train) ./ M ;
mbe2= sum(t_sim2 - T_test) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])
SE1=std(t_sim1-T_train);
RPD1=std(T_train)/SE1;
disp(['剩余预测残差RPD为：  ',num2str(RPD1)])

SE=std(t_sim2-T_test);
RPD2=std(T_test)/SE;
disp(['剩余预测残差RPD为：  ',num2str(RPD2)])




%%
figure
plot(T_train,t_sim1,'*r');
xlabel('真实值')
ylabel('预测值')
string = {'训练集效果图';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
% 获取直线的两个点的坐标
x = h.XData;
y = h.YData;

% 计算斜率
slope = (y(2) - y(1)) / (x(2) - x(1));
intercept = y(1) - slope * x(1);
% %% LOD
% 
% % 收集空白样本的自变量数据，假设为X_blank,样本数在行
% t_sim3 = sim(net, X_blank');
% % T_sim3 = mapminmax('reverse', t_sim3, ps_output);
% 
% S3= mean(t_sim3);
% LOD3 = S3/slope;
% S4=std(t_sim3);
% LOD4=3.3*S4/slope;
% disp(['最低检测限LOD为：', num2str(LOD3)]);
% 
% t_sim33=10.^(t_sim3);
% 
% h0=(mean(t_sim11).^2)/((sum(t_sim11)-mean(t_sim11)).^2);
% var1=var(t_sim33);
% var2=var(t_sim11);
% LOD5=3.3*(sqrt((var1*(1+h0)/(0.979).^2)+(h0*var2)));
%% 验证集
%归一化
% [x_blank1, ps_input] = mapminmax(X_blank1, 0, 1);
t_sim4 = sim(net, X_blank1);
% T_sim4 = mapminmax('reverse', t_sim4, ps_output);
%% 验证集绘图
error4 = sqrt(sum((Y_blank1 - t_sim4).^2)./15);
R4 = 1 - norm(Y_blank1 -  t_sim4)^2 / norm(Y_blank1 -  mean(Y_blank1 ))^2;
mse4 = sum((t_sim4 - Y_blank1).^2)./15;
SE4=std(t_sim4-Y_blank1);
RPD4=std(Y_blank1)/SE4;
figure
plot(1:15,Y_blank1,'r-*',1:15,t_sim4,'b-o','LineWidth',1.5)
legend('真实值','PCA-PLS预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'验证集预测结果对比';['(R^2 =' num2str(R4) ' RMSE= ' num2str(error4)  ' MSE= ' num2str(mse4) ' RPD= ' num2str(RPD4) ')']};
title(string)
disp(['-----------------------验证集误差计算--------------------------'])
disp(['评价结果如下所示：'])
disp(['均方误差MSE为：       ',num2str(mse4)])
disp(['均方根误差RMSEP为：  ',num2str(error4)])
disp(['决定系数R^2为：  ',num2str(R4)])
disp(['剩余预测残差RPD为：  ',num2str(RPD4)])
grid


disp(['-----------------------校准集误差--------------------------'])
disp(['评价结果如下所示：'])
disp(['校准集斜率slope为：       ',num2str(slope)])
disp(['截距intercept为：  ',num2str(intercept)])
disp(['相关系数r为：  ',num2str(r)])
disp(['决定系数R^2为：  ',num2str(R1)])
disp(['均方根误差RMSECV为：  ',num2str(RMSEC1)])
disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['剩余预测残差RPD为：  ',num2str(RPD1)])
disp(['剩余预测偏差bias为：  ',num2str(bias1)])
grid
disp(['-----------------------预测集误差--------------------------'])
disp(['评价结果如下所示：'])
disp(['决定系数R^2为：  ',num2str(R2)])
disp(['均方根误差RMSECV为：  ',num2str(RMSEC2)])
disp(['训练集数据的MAE为：', num2str(mae2)])
disp(['训练集数据的MBE为：', num2str(mbe2)])
disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
disp(['剩余预测偏差bias为：  ',num2str(bias2)])
grid