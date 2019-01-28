MATRIX_M = 100;
MATRIX_N = 100;
MATRIX_NUMBER = 10000;

data = rand(MATRIX_M, MATRIX_N, MATRIX_NUMBER);

tic
for i = 1 : MATRIX_NUMBER
    discrete_cpp(data(:, :, i));
end
toc

tic
for i = 1 : MATRIX_NUMBER
    discrete_matlab(data(:, :, i));
end
toc