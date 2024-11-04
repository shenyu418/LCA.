%% I. 清空环境变量

clear all 
clc
close all
%% 基线校正
% 加载拉曼光谱数据 
data = load('raman_spectrum.txt');
x = data(:,1); 
% 拉曼位移
y = data(:,2);
% 光谱强度 % 选择基线点 % 可以手动选择或使用自动算法，例如寻找峰谷点 
baseline_points = [10 50 100 150 200];
baseline_x = x(baseline_points); 
baseline_y = y(baseline_points);
% 多项式拟合 
order = 3; 
% 多项式阶数 
p = polyfit(baseline_x, baseline_y, order); 
% 计算拟合基线 
baseline = polyval(p, x);
% 基线矫正 
corrected_y = y - baseline;
% 绘图
figure; plot(x, y, 'b', x, baseline, 'r', x, corrected_y, 'g'); 
legend('原始光谱', '拟合基线', '校正后光谱'); 
xlabel('拉曼位移 (cm^{-1})');
ylabel('强度'); 
title('拉曼光谱基线校正');
%% 样本数在列
res= xlsread('原始牛奶验证.xlsx');
res=res';
spectrum1=xlsread('标准图谱.xls');
data_matrix = []; % 初始化数据矩阵
load('data.mat')
A= size(res,2);

%
for i= 1:A
 disp(['第几次循环：       ',num2str(i)])
 %%1为标准拉曼图谱

% 读取数据
data1 = spectrum1;
intensity1 = data1(:, 1);
intensity2 = res(:,i);

% 平滑数据
smoothed_intensity1 = smooth(intensity1, 10);
smoothed_intensity2 = smooth(intensity2, 10);

% 寻找峰值
[pks1, locs1] = findpeaks(smoothed_intensity1, 'MinPeakHeight', 0.02*max(smoothed_intensity1));
[pks2, locs2] = findpeaks(smoothed_intensity2, 'MinPeakHeight', 0.02*max(smoothed_intensity2));

% 找到重叠的峰
overlap_locs = [];
overlap_pks = [];
for m = 1:length(locs2)
    if any(abs(locs2(m) - locs1) <4.5) % 设置一个阈值，判断是否与1中的峰重叠
        overlap_locs = [overlap_locs; locs2(m)];
        overlap_pks = [overlap_pks; pks2(m)];
    end
    
end

if isempty(overlap_locs)
        disp(['没有重叠峰的光谱是：       ',num2str(i)])
        continue;
        % 如果overlap_locs为空集，则跳过剩下的代码执行下一次循环的语句
end

%%修正光谱数据
% first_peak_ratio = pks1(2) / overlap_pks(2);
if A<40
    first_peak_ratio = (pks1(1))/overlap_pks(1) ;
else
    first_peak_ratio = (pks1(2)) / overlap_pks(2) ;
end
intensity2_corrected = smoothed_intensity2;
 for n = 1:length(intensity2_corrected)
%     if ~ismember(intensity2_corrected(i), overlap_locs)
%         intensity2_corrected(i) = intensity1(i) / first_peak_ratio;
%     end
    if ismember(n, overlap_locs)
        continue; % 跳过overlap_locs中的数据
    end
    intensity2_corrected(n) = smoothed_intensity1(n) / (first_peak_ratio);
 end

[pks3, locs3] = findpeaks(intensity2_corrected, 'MinPeakHeight', 0.02*max(intensity2_corrected));



%修正多重峰的峰值
x=intensity2_corrected;
model = createFit(t, x)
newx=model(t);
t=1:length(x);
plot(x)
hold on
plot(newx)


data_matrix = [data_matrix, newx]; % 将最后一列数据添加到数据矩阵中

    if i==length(res(:, 2))
      continue; % 跳过overlap_locs中的数据
    end 

end
%% 保存数据
filename = '牛奶修正验证.xlsx';
sheet = 'Sheet1';
% 指定保存的位置
folder = 'C:\Users\59853\Desktop\数据\数据处理方法\016_基于BP神经网络的多输出数据回归预测\牛奶'; % 替换为你想要保存的文件夹路径"C:\Users\59853\Desktop\数据\PCA-PLS (2)"

% 构建完整的文件路径
fullpath = fullfile(folder, filename);
% 使用xlswrite函数将矩阵数据保存为XLSX文件
xlswrite(fullpath, data_matrix', sheet);


