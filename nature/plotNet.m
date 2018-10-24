clear;clc;

load('../models/net_rl.mat');

params = net.params;

% conv1f = params(1).value;     % 4-D single
% [n1, n2, n3, n4] = size(conv1f);    % 7x7x7x96
% slice()

%% The last full connected layer fc5(except action layer fc6 and condidence fc7)
% fc5 should contains fc5, relu5, drop5, concat layer.
f5cf = params(9).value;     % 4-D   single
f5cb = params(10).value;    % 1x512 single

% plot(f5cb);

[n1, n2, n3, n4] = size(f5cf);  % 1x1x512x512


