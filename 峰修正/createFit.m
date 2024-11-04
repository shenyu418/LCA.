function [fitresult, gof] = createFit(t, x)
%CREATEFIT(T,X)
%  创建拟合
%
%      X 输入: t
%      Y 输出: x
%  输出:
%      fitresult: 拟合结果
%      gof: 拟合优化信息结构体
%

%% 
[xData, yData] = prepareCurveData( t, x );

% 设置fittype
ft = fittype( 'gauss8' );
excludedPoints = excludedata( xData, yData, 'Indices', [658 661] );%排除点
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0 -Inf -Inf 0];
opts.Robust = 'Bisquare';
opts.StartPoint = [4696.151828 403 7.18928581804622 1250.26665683067 399 9.30130299556896 759.1982573 867 16.9689578672152 718.7796448 540 10.6189198952359 659.561370833027 415 11.6373336420272 255.5498633 662 18.1859672976678 244.018158669125 385 19.3770708682917 231.826128097039 553 22.4346360515272];
opts.Exclude = excludedPoints;

% 拟合
[fitresult, gof] = fit( xData, yData, ft, opts );


figure( 'Name', '修正结果' );
h = plot( fitresult, xData, yData, excludedPoints );
legend( h, 'x vs. t', '已排除x vs. t', '修正结果', 'Location', 'NorthEast', 'Interpreter', 'none' );
xlabel( 't', 'Interpreter', 'none' );
ylabel( 'x', 'Interpreter', 'none' );
grid on


