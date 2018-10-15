function [t,p, results] = adnet_demo (vid_path)
% ADNET_DEMO Demonstrate `action-decision network'
% 单目标跟踪，相机会动
% vid_path： 里面包含一个图片文件夹，一个标注文件groundtruth_rect.txt
%           标注文件里的每一行是一个框，包含四个坐标，用逗号分开
% 
% HaiDong Wang, 2018年10月15日.

if nargin < 1       %　当没有参数的时候，跟踪的视频数据从data/Freeman1中进行读取
    vid_path = 'data/Freeman1';
end

addpath('test/');
addpath(genpath('utils/'));     % 增加utils目录以及其子目录下的路径

init_settings;

run(matconvnet_path);

load('models/net_rl.mat');

opts.visualize = true;
opts.printscreen = true;

rng(1004);
[results, t, p] = adnet_test(net, vid_path, opts);
fprintf('precision: %f, fps: %f\n', p(20), size(results, 1)/t);

