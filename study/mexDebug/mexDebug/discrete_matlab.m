function [ B ] = discrete_matlab( A )
    [M, N] = size(A);
    B = zeros(M, N);
    for i = 1 : M
        for j = 1 : N
            if A(i, j) < 1/3
                B(i, j) = 0;
            elseif A(i, j) < 2/3
                B(i, j) = 1;
            else
                B(i, j) = 2;
            end
        end
    end
end