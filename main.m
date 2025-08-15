close all; clearvars; clc;
rng(0);

%% Parameters
rho = .1;
eps = 1e-1;
T = 10;
n = 2; m = 2; p = 2;
A = .1 * [1 1; 0 1];
B = eye(m);
C = eye(p);
Q = eye(n);
R = eye(m);
S = eye(n);
% nominal covariances
X0h = random_psd_matrix(n, 1, 2);
Wh = cell(T,1);
Vh = cell(T,1);
for t = 1:T
    Wh{t} = random_psd_matrix(n, 1, 2);
    Vh{t} = random_psd_matrix(p, 1, 2);
end

% check if the ambiguity sets are possibly empty
empty = false;

if trace(X0h + S) > rho
    empty = true;
end
for t = 1:T
    if trace(Wh{t} + S) > rho || trace(Vh{t} + S) > rho
        empty = true;
    end
end

% preallocate for stacked system dynamics
H = zeros(n*(T+1), m*T);
G = zeros(n*(T+1), n*(T+1));

% fill strictly lower‐triangular blocks
for i = 2:T+1 
    for j = 1:(i-1)
        power = i-j;
        if power==1
            Apow = A;
        else
            Apow = A^power;
        end
        row_idx = (i-1)*n + (1:n);
        col_idx = (j-1)*m + (1:m);
        H(row_idx, col_idx) = Apow * B;
    end
end

for i = 0:T
    for j = 0:i
        block = A^(i-j);
        rows = i*n + (1:n);
        cols = j*n + (1:n);
        G(rows,cols) = block;
    end
end

% cost over the horizon
QT = kron(eye(T+1), Q);
RT = kron(eye(T), R);

pattern = zeros(T, T+1);
for i = 1:T
    pattern(i, i) = 1;
end
CT = kron(pattern, C); 

D = CT * G;

cost = Wasserstein_SDP(T, n, m, p, rho, X0h, Wh, Vh, H, G, D, QT, RT);
disp(cost)
cost = Sinkhorn_conic(T, n, m, p, rho, eps, X0h, Wh, Vh, H, G, D, QT, RT, S);
if cost == inf && empty 
    error("Unfeasible problem. Increase the radius or decrease the regularizer...")
end
disp(cost)

function A = random_psd_matrix(n, lambda_min, lambda_max)
% Generate a symmetric positive definite matrix with eigenvalues in [lambda_min, lambda_max]
    Q = orth(randn(n));
    D = diag(lambda_min + (lambda_max - lambda_min) * rand(n,1));
    A = Q * D * Q';
end
