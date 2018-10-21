clear;

load('../models/net_rl.mat');

params = net.params;

conv1f = params(1).value;     % 4-D single
[n1, n2, n3, n4] = size(conv1f);    % 7x7x7x96
% slice()