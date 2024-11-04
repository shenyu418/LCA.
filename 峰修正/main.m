%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res= xlsread('50�鴿������2.xlsx');
res=res';
X_blank = xlsread('jidi.xlsx');
Y = xlsread('outputd2');
X_blank1 =xlsread('��֤');
X_blank1=X_blank1';
Y_blank1=xlsread('��֤output');
Y_blank1=Y_blank1';

% %%  ����ѵ�����Ͳ��Լ�
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
    temp = randperm(10);%1��������У�2���������
%ѵ������
    P_train1= X1(temp(1:7),:);%ð�Ŵ���ȡ���������л������У�'����ת��
    P_test1 = X1(temp(8:end),:);
    
% ���Լ���
    T_train1= Y1(temp(1:7),:);
    T_test1 = Y1(temp(8:end),:);
    
    % ��P_train1����ƴ�ӵ�P_train
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
% %% CARS��ά
% 
% egg_y = [ P_train;P_test];
% egg_labels_y = [T_train;T_test];
% 
% % ��ȡ�������Σ�SelectedVariables�ǵõ��ı������б��
% %MCCV=plsmccv(egg_y,egg_labels_y,100,'center',50,0.9,1);
% sCARS=scarspls(egg_y,egg_labels_y,100,10,'center',10);% CARS�ļ򻯰汾
% % scarspls.mat�������ֱ���ѡ����
% plotcars(sCARS);
% SelectedVariables=sCARS.vsel;
% % �õ�CARS��Ĳ��ξ���
% CARS_labels = SelectedVariables;
%  P_train =  P_train(:,CARS_labels);
% P_test = P_test(:,CARS_labels);


   P_train =P_train'; 
    P_test=P_test';
    T_train=T_train';
    T_test=T_test';
M = size(P_train,2);
N = size(P_test,2);



%%  ��������
net = newff(P_train, T_train, 10);
%%  ����ѵ������
net.trainParam.epochs = 1000;     % �������� 1000
net.trainParam.goal = 1e-6;       % �����ֵ1e-6
net.trainParam.lr = 0.001;         % ѧϰ��0.01
net.trainFcn = 'trainlm';

%%  ѵ������ ��Ҫ����������
net = train(net, P_train, T_train);

%%  ������� ���������У�ѵ����
t_sim1 = sim(net, P_train);
t_sim2 = sim(net, P_test );

%%  ���������
error1 = sqrt(sum((t_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((t_sim2 - T_test ).^2) ./ N);
RMSEC1 = sqrt(sum((t_sim1 - T_train).^2) ./ (M-1));
RMSEC2 = sqrt(sum((t_sim2 - T_test).^2) ./ (N-1));
disp(['���������RMSEPΪ��  ',num2str(error2)])
disp(['���������RMSECVΪ��  ',num2str(error1)])

%%  ��ͼ
figure
subplot(2, 1, 1)
plot(1: M, T_train, 'r-*', 1: M, t_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])

grid

subplot(2, 1, 2)
plot(1: N, T_test, 'r-*', 1: N, t_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid
figure;
plotregression(T_test,t_sim2,['�ع�ͼ']);

%%  �ָ���
disp('**************************')
disp(['���������', num2str(i)])
disp('**************************')

%%  ���ָ�����
%���ϵ��
covariance_XY = sum((T_train - mean(T_train)) .* (((t_sim1)) - mean(t_sim1))) / (length(T_train)- 1);
std_X = sqrt(sum((T_train - mean(T_train)).^2) / (length(T_train) - 1));
std_Y = sqrt(sum((t_sim1 - mean(t_sim1)).^2) / (length(t_sim1) - 1));
r = covariance_XY / (std_X * std_Y  );
t_sim11=10.^(t_sim1);
T_train1=10.^(T_train);
bias1= mean(T_train - t_sim1);
bias2= mean(T_test - t_sim2);
% ����ϵ�� R2
R1 = 1 - sum((T_train - t_sim1).^2) / sum((T_train - mean(T_train)).^2);
R2 = 1 - sum((T_test  - t_sim2).^2 )/ sum((T_test  - mean(T_test )).^2);


disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

% ƽ��������� MAE
mae1 = sum(abs(t_sim1 - T_train)) ./ M ;
mae2 = sum(abs(t_sim2 - T_test)) ./ N ;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])

% ƽ�������� MBE
mbe1= sum(t_sim1 - T_train) ./ M ;
mbe2= sum(t_sim2 - T_test) ./ N ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])
SE1=std(t_sim1-T_train);
RPD1=std(T_train)/SE1;
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD1)])

SE=std(t_sim2-T_test);
RPD2=std(T_test)/SE;
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD2)])




%%
figure
plot(T_train,t_sim1,'*r');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string = {'ѵ����Ч��ͼ';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
% ��ȡֱ�ߵ������������
x = h.XData;
y = h.YData;

% ����б��
slope = (y(2) - y(1)) / (x(2) - x(1));
intercept = y(1) - slope * x(1);
% %% LOD
% 
% % �ռ��հ��������Ա������ݣ�����ΪX_blank,����������
% t_sim3 = sim(net, X_blank');
% % T_sim3 = mapminmax('reverse', t_sim3, ps_output);
% 
% S3= mean(t_sim3);
% LOD3 = S3/slope;
% S4=std(t_sim3);
% LOD4=3.3*S4/slope;
% disp(['��ͼ����LODΪ��', num2str(LOD3)]);
% 
% t_sim33=10.^(t_sim3);
% 
% h0=(mean(t_sim11).^2)/((sum(t_sim11)-mean(t_sim11)).^2);
% var1=var(t_sim33);
% var2=var(t_sim11);
% LOD5=3.3*(sqrt((var1*(1+h0)/(0.979).^2)+(h0*var2)));
%% ��֤��
%��һ��
% [x_blank1, ps_input] = mapminmax(X_blank1, 0, 1);
t_sim4 = sim(net, X_blank1);
% T_sim4 = mapminmax('reverse', t_sim4, ps_output);
%% ��֤����ͼ
error4 = sqrt(sum((Y_blank1 - t_sim4).^2)./15);
R4 = 1 - norm(Y_blank1 -  t_sim4)^2 / norm(Y_blank1 -  mean(Y_blank1 ))^2;
mse4 = sum((t_sim4 - Y_blank1).^2)./15;
SE4=std(t_sim4-Y_blank1);
RPD4=std(Y_blank1)/SE4;
figure
plot(1:15,Y_blank1,'r-*',1:15,t_sim4,'b-o','LineWidth',1.5)
legend('��ʵֵ','PCA-PLSԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'��֤��Ԥ�����Ա�';['(R^2 =' num2str(R4) ' RMSE= ' num2str(error4)  ' MSE= ' num2str(mse4) ' RPD= ' num2str(RPD4) ')']};
title(string)
disp(['-----------------------��֤��������--------------------------'])
disp(['���۽��������ʾ��'])
disp(['�������MSEΪ��       ',num2str(mse4)])
disp(['���������RMSEPΪ��  ',num2str(error4)])
disp(['����ϵ��R^2Ϊ��  ',num2str(R4)])
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD4)])
grid


disp(['-----------------------У׼�����--------------------------'])
disp(['���۽��������ʾ��'])
disp(['У׼��б��slopeΪ��       ',num2str(slope)])
disp(['�ؾ�interceptΪ��  ',num2str(intercept)])
disp(['���ϵ��rΪ��  ',num2str(r)])
disp(['����ϵ��R^2Ϊ��  ',num2str(R1)])
disp(['���������RMSECVΪ��  ',num2str(RMSEC1)])
disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD1)])
disp(['ʣ��Ԥ��ƫ��biasΪ��  ',num2str(bias1)])
grid
disp(['-----------------------Ԥ�⼯���--------------------------'])
disp(['���۽��������ʾ��'])
disp(['����ϵ��R^2Ϊ��  ',num2str(R2)])
disp(['���������RMSECVΪ��  ',num2str(RMSEC2)])
disp(['ѵ�������ݵ�MAEΪ��', num2str(mae2)])
disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe2)])
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD2)])
disp(['ʣ��Ԥ��ƫ��biasΪ��  ',num2str(bias2)])
grid