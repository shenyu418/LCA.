%% I. ��ջ�������

clear all 
clc
close all
%% ����У��
% ���������������� 
data = load('raman_spectrum.txt');
x = data(:,1); 
% ����λ��
y = data(:,2);
% ����ǿ�� % ѡ����ߵ� % �����ֶ�ѡ���ʹ���Զ��㷨������Ѱ�ҷ�ȵ� 
baseline_points = [10 50 100 150 200];
baseline_x = x(baseline_points); 
baseline_y = y(baseline_points);
% ����ʽ��� 
order = 3; 
% ����ʽ���� 
p = polyfit(baseline_x, baseline_y, order); 
% ������ϻ��� 
baseline = polyval(p, x);
% ���߽��� 
corrected_y = y - baseline;
% ��ͼ
figure; plot(x, y, 'b', x, baseline, 'r', x, corrected_y, 'g'); 
legend('ԭʼ����', '��ϻ���', 'У�������'); 
xlabel('����λ�� (cm^{-1})');
ylabel('ǿ��'); 
title('�������׻���У��');
%% ����������
res= xlsread('ԭʼţ����֤.xlsx');
res=res';
spectrum1=xlsread('��׼ͼ��.xls');
data_matrix = []; % ��ʼ�����ݾ���
load('data.mat')
A= size(res,2);

%
for i= 1:A
 disp(['�ڼ���ѭ����       ',num2str(i)])
 %%1Ϊ��׼����ͼ��

% ��ȡ����
data1 = spectrum1;
intensity1 = data1(:, 1);
intensity2 = res(:,i);

% ƽ������
smoothed_intensity1 = smooth(intensity1, 10);
smoothed_intensity2 = smooth(intensity2, 10);

% Ѱ�ҷ�ֵ
[pks1, locs1] = findpeaks(smoothed_intensity1, 'MinPeakHeight', 0.02*max(smoothed_intensity1));
[pks2, locs2] = findpeaks(smoothed_intensity2, 'MinPeakHeight', 0.02*max(smoothed_intensity2));

% �ҵ��ص��ķ�
overlap_locs = [];
overlap_pks = [];
for m = 1:length(locs2)
    if any(abs(locs2(m) - locs1) <4.5) % ����һ����ֵ���ж��Ƿ���1�еķ��ص�
        overlap_locs = [overlap_locs; locs2(m)];
        overlap_pks = [overlap_pks; pks2(m)];
    end
    
end

if isempty(overlap_locs)
        disp(['û���ص���Ĺ����ǣ�       ',num2str(i)])
        continue;
        % ���overlap_locsΪ�ռ���������ʣ�µĴ���ִ����һ��ѭ�������
end

%%������������
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
        continue; % ����overlap_locs�е�����
    end
    intensity2_corrected(n) = smoothed_intensity1(n) / (first_peak_ratio);
 end

[pks3, locs3] = findpeaks(intensity2_corrected, 'MinPeakHeight', 0.02*max(intensity2_corrected));



%�������ط�ķ�ֵ
x=intensity2_corrected;
model = createFit(t, x)
newx=model(t);
t=1:length(x);
plot(x)
hold on
plot(newx)


data_matrix = [data_matrix, newx]; % �����һ��������ӵ����ݾ�����

    if i==length(res(:, 2))
      continue; % ����overlap_locs�е�����
    end 

end
%% ��������
filename = 'ţ��������֤.xlsx';
sheet = 'Sheet1';
% ָ�������λ��
folder = 'C:\Users\59853\Desktop\����\���ݴ�����\016_����BP������Ķ�������ݻع�Ԥ��\ţ��'; % �滻Ϊ����Ҫ������ļ���·��"C:\Users\59853\Desktop\����\PCA-PLS (2)"

% �����������ļ�·��
fullpath = fullfile(folder, filename);
% ʹ��xlswrite�������������ݱ���ΪXLSX�ļ�
xlswrite(fullpath, data_matrix', sheet);


