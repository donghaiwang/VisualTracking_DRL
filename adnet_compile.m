function adnet_compile
% ADNET_COMPILE 
% 
% HaiDong Wang, 2018.
if ispc
    cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0';   % windows下编译暂时有问题
else
    cudaRoot = '/usr/local/cuda-8.0';  % ubuntu中cuda的安装目录（指定使用cuda 8.0，其他版本的cuda可能有问题）
end

fprintf('compile matconvnet\n');
run matconvnet/matlab/vl_setupnn.m
cd matconvnet/
vl_compilenn('enableGpu', true, ...
    'cudaRoot', cudaRoot, ...
    'cudaMethod', 'nvcc');
% vl_compilenn('enableGpu', true);
cd ..

cd utils/
addpath cropRectanglesMex
if ispc
    run build_cropRectanglesMex_on_windows.m
else
    run build_cropRectanglesMex.m
end
cd ..

