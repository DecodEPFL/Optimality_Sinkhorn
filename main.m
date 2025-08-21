close all; clearvars; clc;
rng(1234);

%% Parameters
rho = 10;
eps = 1e-1;
T = 10;
n = 2; m = 2; p = 2;
A = [1.1 0.1 ; 0 .95];
B = eye(m);
C = eye(p);
Q = 10*eye(n);
R = eye(m);
S = eye(n);

% nominal covariances
Wh = cell(T,1);
Vh = cell(T,1);
for t = 1:T
    Wh{t} = .1 * random_psd_matrix(n, 1, 2);
    Vh{t} = .1 * random_psd_matrix(p, 1, 2);
end
X0h = .1 * random_psd_matrix(n, 1, 2);

% check if the ambiguity sets are possibly empty
% empty = false;
% 
% for t = 1:T
%     if trace(Wh{t} + S) > rho || trace(Vh{t} + S) > rho
%         empty = true;
%     end
% end

bar_rho_x0 = (eps/2) * ( trace( S \ (X0h + (eps/2)*eye(n)) ) - n + log(det(S)) - n * log(eps/2) );
if rho < bar_rho_x0
    error("Unfeasible problem. Increase the radius or decrease the regularizer...")
end

for t=1:T
    bar_rho_w = (eps/2) * ( trace( S \ (Wh{t} + (eps/2)*eye(n)) ) - n + log(det(S)) - n * log(eps/2) );
    bar_rho_v = (eps/2) * ( trace( S \ (Vh{t} + (eps/2)*eye(n)) ) - n + log(det(S)) - n * log(eps/2) );
    if rho < bar_rho_w || rho < bar_rho_v
        error("Unfeasible problem. Increase the radius or decrease the regularizer...")
    end
end

% preallocate for stacked system dynamics
H = zeros(n*(T+1), m*T);
G = zeros(n*(T+1), n*(T+1));

% fill strictly lower‐triangular blocks
for i = 2:T+1
    for j = 1:(i-1)
        H((i-1)*n + (1:n), (j-1)*m + (1:m)) = A^(i-j-1) * B;
    end
end

for i = 0:T
    for j = 0:i
        block = A^(i-j);
        G(i*n + (1:n),j*n + (1:n)) = block;
    end
end

% cost over the horizon
QT = kron(eye(T+1), Q);
RT = kron(eye(T), R);

% stack of the matrix C
pattern = zeros(T, T+1);
for i = 1:T
    pattern(i, i) = 1;
end
CT = kron(pattern, C); 

D = CT * G;

% [cost_w, Ww, Vw] = Wasserstein_SDP(T, n, m, p, rho, X0h, Wh, Vh, H, G, D, QT, RT);
% disp(costW)
[cost_s, Ws, Vs] = Sinkhorn_conic(T, n, m, p, rho, eps, X0h, Wh, Vh, H, G, D, QT, RT, S);
% disp(costS)

%% SLS formulation to get the maps from noise to state and input 
sls.A = kron(eye(T+1), A);
sls.B = kron(eye(T+1), B);
sls.B = sls.B(:, 1:T*m);
sls.C = CT;
sls.I = eye(n*(T+1));
sls.Z = [zeros(n, n*(T)) zeros(n, n); eye(n*(T)) zeros(n*T, n)];

[Phi_xx_nom, Phi_xy_nom, Phi_ux_nom, Phi_uy_nom, cost_nom] = causal_unconstrained_h2(m, p, T, blkdiag(QT, RT), sls, blkdiag(X0h, Wh{:}, Vh{:}));
% [Phi_xxW, Phi_xyW, Phi_uxW, Phi_uyW, costW] = causal_unconstrained_h2(m, p, T, blkdiag(QT, RT), sls, blkdiag(Ww, Vw));
[Phi_xxS, Phi_xyS, Phi_uxS, Phi_uyS, costS] = causal_unconstrained_h2(m, p, T, blkdiag(QT, RT), sls, blkdiag(Ws, Vs));

Phi_nom = [Phi_xx_nom, Phi_xy_nom; Phi_ux_nom, Phi_uy_nom];
Phi_S = [Phi_xxS, Phi_xyS; Phi_uxS, Phi_uyS];

%% Test the policies on-sample
mc_tests = 5000;
nom = struct();
rob = struct();
for i=1:mc_tests
    % sample from the nominal Gaussian
    delta = mvnrnd(zeros(T*(n+p) + n, 1), blkdiag(X0h, Wh{:}, Vh{:}))';   % column vector sample
    tmp = delta' * Phi_nom' * blkdiag(QT, RT) * Phi_nom * delta;
    tmp1 = delta' * Phi_S' * blkdiag(QT, RT) * Phi_S * delta;
    nom.cost_on(i) = tmp;
    rob.cost_on(i) = tmp1;
end

mu_nom = mean(nom.cost_on);
mu_rob = mean(rob.cost_on);

figure;
h1 = histogram(nom.cost_on, 100);
hold on
h2 = histogram(rob.cost_on, 100);
% Imposta colore e trasparenza
h1.FaceColor = 'b';    
h1.FaceAlpha = 0.8;    
h2.FaceColor = 'r';    
h2.FaceAlpha = 0.2;    

% Add vertical lines at the means
yl = ylim; % current y-axis limits
plot([mu_nom mu_nom], yl, 'b--','LineWidth',2);
plot([mu_rob mu_rob], yl, 'r--','LineWidth',2);
% set(gca,'XScale','log');

%% Test policies off-sample
for i=1:mc_tests
    % sample from the worst-case Gaussian
    delta = mvnrnd(zeros(T*(n+p) + n, 1), blkdiag(Ws, Vs))';   % column vector sample
    tmp = delta' * Phi_nom' * blkdiag(QT, RT) * Phi_nom * delta;
    tmp1 = delta' * Phi_S' * blkdiag(QT, RT) * Phi_S * delta;
    nom.cost_off(i) = tmp;
    rob.cost_off(i) = tmp1;
end

mu_nom = mean(nom.cost_off);
mu_rob = mean(rob.cost_off);

figure;
h1 = histogram(nom.cost_off, 100);
hold on
h2 = histogram(rob.cost_off, 100);
% Imposta colore e trasparenza
h1.FaceColor = 'b';    
h1.FaceAlpha = 0.8;    
h2.FaceColor = 'r';    
h2.FaceAlpha = 0.2;    

% Add vertical lines at the means
yl = ylim; % current y-axis limits
plot([mu_nom mu_nom], yl, 'b--','LineWidth',2);
plot([mu_rob mu_rob], yl, 'r--','LineWidth',2);
set(gca,'XScale','log');

%% Test policies on unseen noise realizations
for i=1:mc_tests

    W_new = cell(T,1);
    V_new = cell(T,1);

    % sample new covariances
    W_1 = random_psd_matrix(n, 0, 1);
    for t=1:T
        W_new{t} = random_psd_matrix(n, 0, 1);
        V_new{t} = random_psd_matrix(p, 0, 1);
    end

    delta = mvnrnd(zeros(T*(n+p) + n, 1), blkdiag(W_1, W_new{:}, V_new{:}))';   % column vector sample
    tmp = delta' * Phi_nom' * blkdiag(QT, RT) * Phi_nom * delta;
    tmp1 = delta' * Phi_S' * blkdiag(QT, RT) * Phi_S * delta;
    nom.cost_unseen(i) = tmp;
    rob.cost_unseen(i) = tmp1;
end

mu_nom = mean(nom.cost_unseen);
mu_rob = mean(rob.cost_unseen);

figure;
h1 = histogram(nom.cost_unseen, 100);
hold on
h2 = histogram(rob.cost_unseen, 100);
% Imposta colore e trasparenza
h1.FaceColor = 'b';    
h1.FaceAlpha = 0.8;    
h2.FaceColor = 'r';    
h2.FaceAlpha = 0.2;    

% Add vertical lines at the means
yl = ylim; % current y-axis limits
plot([mu_nom mu_nom], yl, 'b--','LineWidth',2);
plot([mu_rob mu_rob], yl, 'r--','LineWidth',2);
set(gca,'XScale','log');

%% Auxiliary functions
function A = random_psd_matrix(n, lambda_min, lambda_max)
% Generate a symmetric positive definite matrix with eigenvalues in [lambda_min, lambda_max]
    Q = orth(randn(n));
    D = diag(lambda_min + (lambda_max - lambda_min) * rand(n,1));
    A = Q * D * Q';
end

function [Phi_xx, Phi_xy, Phi_ux, Phi_uy, cost] = causal_unconstrained_h2(m, p, T, C, sls, Sigma)

    % Define the decision variables of the optimization problem                    
    Phi_uy = sdpvar(m * T, p * T, 'full');
    Phi_ux = Phi_uy * sls.C / (sls.I - sls.Z * sls.A);
    Phi_xy = (sls.I - sls.Z * sls.A) \ (sls.Z * sls.B * Phi_uy);
    Phi_xx = (sls.I - sls.Z * sls.A) \ (sls.I + sls.Z * sls.B * Phi_ux);
    
    % Define the objective function
    objective = norm(sqrtm(C)*[Phi_xx Phi_xy; Phi_ux Phi_uy]*sqrtm(Sigma), 'fro')^2;

    constraints = [];

    % Impose the causal sparsities on the closed loop responses
    for i = 1:T
        for j = (i+1):T   % future columns
            row_idx = (i-1)*m+1 : i*m;
            col_idx = (j-1)*p+1 : j*p;
            constraints = [constraints, Phi_uy(row_idx, col_idx) == 0];
        end
    end

    % Solve the optimization problem
    options = sdpsettings('verbose', 0, 'solver', 'mosek');
    sol = optimize(constraints, objective, options);
    if ~(sol.problem == 0)
        error('Something went wrong...');
    end
    
    % Extract the closed-loop responses corresponding to a unconstrained causal 
    % linear controller that is optimal either in the H2 or in the Hinf sense
    Phi_xx = value(Phi_xx); 
    Phi_xy = value(Phi_xy);
    Phi_ux = value(Phi_ux);
    Phi_uy = value(Phi_uy);
    cost = value(objective);
end
