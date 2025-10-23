close all; clearvars; clc;
rng(1234);
% g = [0.4660, 0.6740, 0.1880];
% r = [0.8500, 0.3250, 0.0980];
%% Parameters
rho_0 = 20;
rho_w = 0.2;
rho_v = 20;
% rho = 1e-7;
eps = 1e-2;
T = 25;
n = 2; m = 1; p = 2;
A = [1.1 0.1 ; 0 1.1];
B = [1; 1];
C = [1 0; 0 1];
Q = eye(n);
R = eye(m);
S = eye(n);

% nominal covariances
Wh = cell(T,1);
Vh = cell(T,1);
for t = 1:T
    Wh{t} = eye(n);
    Vh{t} = .1^2*eye(p);
end
X0h = eye(n);

% check if the ambiguity sets are possibly empty
bar_rho_x0 = (eps/2) * ( trace( S \ (X0h + (eps/2)*eye(n)) ) - n + log(det(S)) - n * log(eps/2) );
if rho_0 < bar_rho_x0
    error("Unfeasible problem. Increase the radius rho_0 or decrease the regularizer...")
end

for t=1:T
    bar_rho_w = (eps/2) * ( trace( S \ (Wh{t} + (eps/2)*eye(n)) ) - n + log(det(S)) - n * log(eps/2) );
    bar_rho_v = (eps/2) * ( trace( S \ (Vh{t} + (eps/2)*eye(n)) ) - n + log(det(S)) - n * log(eps/2) );
    if rho_w < bar_rho_w || rho_v < bar_rho_v
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
% K = compute_finite_horizon_lqr_gains(A, B, Q, R, T);
% L = finite_horizon_kalman_gains(A, C, Wh, Vh, X0h, T);
% x0 = zeros(n, 1);
% % U_LQG = compute_finite_horizon_lqr_gains(A, B, Q, R, T);
% cost = 0;
% for i = 1:10000
%     [~, ~, cost_t] = simulate_lqg(T, A, B, C, K, L, x0, Wh, Vh, Q, R);
%     cost = cost + cost_t;
% end
% cost = cost/10000;
% [cost_w, Ww, Vw, M] = Wasserstein_SDP(T, n, m, p, rho, X0h, Wh, Vh, H, G, D, QT, RT);
[cost_s, Ws, Vs] = Sinkhorn_conic(T, n, m, p, rho_0, rho_w, rho_v, eps, X0h, Wh, Vh, H, G, D, QT, RT, S);
% U_W = -(RT + H'*QT *H) \ (H'*QT*G*Ww*D' + M/2) / (D * Ww * D' + Vw);
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
% Phi_W = [Phi_xxW, Phi_xyW; Phi_uxW, Phi_uyW];

mc_tests = 5000;
nom = struct();
rob = struct();

%% Worst-case performance
load U_LQG25B.mat
[~, W_wc, V_wc] = compute_WC_LQG(U_LQG, H, G, D, X0h, Wh, Vh, T, n, p, rho_0, rho_w, rho_v, QT, RT, S, eps);

for i=1:mc_tests
    % sample from the worst-case Gaussian
    % delta = mvnrnd(zeros(T*(n+p) + n, 1), blkdiag(Ww, Vw))';
    delta = mvnrnd(zeros(T*(n+p) + n, 1), blkdiag(Ws, Vs))';
    delta_wc = mvnrnd(zeros(T*(n+p) + n, 1), blkdiag(W_wc, V_wc))';
    tmp = delta_wc' * Phi_nom' * blkdiag(QT, RT) * Phi_nom * delta_wc;
    tmp1 = delta' * Phi_S' * blkdiag(QT, RT) * Phi_S * delta;
    % tmp1 = delta' * Phi_W' * blkdiag(QT, RT) * Phi_W * delta;
    nom.cost_wc(i) = tmp;
    rob.cost_wc(i) = tmp1;
end

mu_nom = norm(sqrtm(blkdiag(QT, RT)) * Phi_nom*sqrtm(blkdiag(W_wc, V_wc)), 'fro')^2;
mu_rob = norm(sqrtm(blkdiag(QT, RT)) * Phi_S*sqrtm(blkdiag(Ws, Vs)), 'fro')^2;

% Define the figure dimensions in inches
width = 3.5; % One column width
height = 3.5 * 0.35; % Aspect ratio of 4:3, adjust as needed

% Create the figure
figure('Units', 'inches', 'Position', [1, 1, width, height]);

% Define common edges spanning both vectors
edges = linspace(min([nom.cost_wc, rob.cost_wc]), max([nom.cost_wc , rob.cost_wc]), 100+1);

h1 = histogram(nom.cost_wc, edges);
hold on
h2 = histogram(rob.cost_wc, edges);
% Imposta colore e trasparenza
h1.FaceColor = 'r';    
h1.FaceAlpha = 0.5;    
h2.FaceColor = 'g';    
h2.FaceAlpha = 0.5;    

% Add vertical lines at the means
% maxCount = max(max(h1.Values), max(h2.Values));       % maximum bin count
% ylim([0, 1+maxCount]);
xlim([0, prctile([nom.cost_wc rob.cost_wc], 99)]);

p1 = plot([mu_nom mu_nom], ylim, 'r:', 'LineWidth', 2);
p2 = plot([mu_rob mu_rob], ylim, 'g:', 'LineWidth', 2);
ylabel('Frequency', 'Interpreter', 'latex', 'FontSize', 8);
xlabel('$J(\pi, \delta)$', 'Interpreter', 'latex', 'FontSize', 8);
ax = gca;   
ax.FontSize = 6;
ax.XAxis.Exponent = 3;  % force scientific notation with ×10^3
ax.TickLabelInterpreter = 'tex';  % ensures nice formatting
% Legend
legend([h1 h2], {'LQG', 'Sinkhorn DR LQG'}, 'Interpreter', 'latex', 'Location','best', 'FontSize', 6);
% set(gca,'YScale','log');
set(gcf, 'PaperPositionMode', 'auto');
exportgraphics(gcf, 'worst_policy.pdf', 'ContentType', 'vector');

%% Test the policies on-sample

for i=1:mc_tests
    % sample from the nominal Gaussian
    delta = mvnrnd(zeros(T*(n+p) + n, 1), blkdiag(X0h, Wh{:}, Vh{:}))';   % column vector sample
    tmp = delta' * Phi_nom' * blkdiag(QT, RT) * Phi_nom * delta;
    tmp1 = delta' * Phi_S' * blkdiag(QT, RT) * Phi_S * delta;
    % tmp1 = delta' * Phi_W' * blkdiag(QT, RT) * Phi_W * delta;
    nom.cost_on(i) = tmp;
    rob.cost_on(i) = tmp1;
end

mu_nom = norm(sqrtm(blkdiag(QT, RT)) * Phi_nom*sqrtm(blkdiag(X0h, Wh{:}, Vh{:})), 'fro')^2;
mu_rob = norm(sqrtm(blkdiag(QT, RT)) * Phi_S*sqrtm(blkdiag(X0h, Wh{:}, Vh{:})), 'fro')^2;
gain_on = 100 * abs(mu_nom - mu_rob)/mu_nom;

% Define the figure dimensions in inches
width = 3.5; % One column width
height = 3.5 * 0.35; % Aspect ratio of 4:3, adjust as needed

% Create the figure
figure('Units', 'inches', 'Position', [1, 1, width, height]);

% Define common edges spanning both vectors
edges = linspace(min([nom.cost_on, rob.cost_on]), max([nom.cost_on , rob.cost_on]), 100+1);

h1 = histogram(nom.cost_on, edges);
hold on
h2 = histogram(rob.cost_on, edges);
% Imposta colore e trasparenza
h1.FaceColor = 'r';    
h1.FaceAlpha = 0.5;    
h2.FaceColor = 'g';    
h2.FaceAlpha = 0.5;    

% Add vertical lines at the means
maxCount = max(max(h1.Values), max(h2.Values));       % maximum bin count
ylim([0, 1+maxCount]);
% xlim([0, prctile([nom.cost_wc rob.cost_wc], 99)]);

p1 = plot([mu_nom mu_nom], ylim, 'r:', 'LineWidth', 2);
p2 = plot([mu_rob mu_rob], ylim, 'g:', 'LineWidth', 2);
ylabel('Frequency', 'Interpreter', 'latex', 'FontSize', 8);
xlabel('$J(\pi, \delta)$', 'Interpreter', 'latex', 'FontSize', 8);
ax = gca;   
ax.FontSize = 6;
ax.XAxis.Exponent = 3;  % force scientific notation with ×10^3
ax.TickLabelInterpreter = 'tex';  % ensures nice formatting
% set(gca,'XScale','log');
set(gcf, 'PaperPositionMode', 'auto');
% Legend
legend([h1 h2], {'LQG', 'Sinkhorn DR LQG'}, 'Interpreter', 'latex', 'Location','best', 'FontSize', 6);
xlim([0, prctile([nom.cost_on rob.cost_on], 99)]);
exportgraphics(gcf, 'on_policy.pdf', 'ContentType', 'vector');

%% Auxiliary functions
function A = random_psd_matrix(n, lambda_min, lambda_max)
% Generate a symmetric positive definite matrix with eigenvalues in [lambda_min, lambda_max]
    Q = orth(randn(n));
    D = diag(lambda_min + (lambda_max - lambda_min) * rand(n,1));
    A = Q * D * Q';
end

function [Phi_xx, Phi_xy, Phi_ux, Phi_uy, cost] = causal_unconstrained_h2(m, p, T, C, sls, Sigma)
% This function computes the H2 controller based on the given covariance
% matrix Sigma
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

function [cost, W_opt, V_opt] = compute_WC_LQG(U_star, H, G, D, X0h, Wh, Vh, T, n, p, rho_0, rho_w, rho_v, Q, R, S, eps)
% This function computes the WC covariances in the Sinkhorn ambiguity set
% given the optimal feedback U_star of the LQG nominal problem
    % Decision variables
    for t = 1:T+1
        Wblocks{t} = sdpvar(n,n);
    end
    W = blkdiag(Wblocks{:});
    for t = 1:T
        Vblocks{t} = sdpvar(p,p);
    end
    V  = blkdiag(Vblocks{:});

    Ex0 = sdpvar(n,n);
    Ew  = sdpvar(n,n,T);
    Ev  = sdpvar(p,p,T);
    
    % Constraints
    % Positive definitiness 
    Fcon = [Ex0 >= 0, Wblocks{1} >= 0];
    
    for t = 1:T
        Fcon = [Fcon, Ew(:,:,t) >= 0, Ev(:,:,t) >= 0, Vblocks{t} >= 0, Wblocks{t+1} >= 0];
    end
    
    % Linearization using Schur's complement of the nonlinear part: Ez^2 <= \hat{Z}^1/2 Z \hat{Z}^1/2
    Fcon = [Fcon, [sqrtm(X0h) * Wblocks{1} * sqrtm(X0h) + eps^2/16 * eye(n), Ex0; Ex0, eye(n)] >= 0];
    
    for t=1:T
      tmp1 = sqrtm(Wh{t});
      tmp2 = sqrtm(Vh{t});
      Fcon = [Fcon, [tmp1 * Wblocks{t+1} * tmp1 + eps^2/16 * eye(n), Ew(:,:,t); Ew(:,:,t), eye(n)] >= 0];
      Fcon = [Fcon, [tmp2 * Vblocks{t} * tmp2 + eps^2/16 * eye(p), Ev(:,:,t); Ev(:,:,t), eye(p)] >= 0];
    end

    Fcon = [Fcon, trace( Wblocks{1} + .5 * eps * (S \ Wblocks{1}) - 2*Ex0 ) - eps / 2 * n * log(geomean(Ex0 - eps/4*eye(n))) <= ...
                  rho_0 - trace(X0h) - eps/2 * (logdet(S) + logdet(X0h)) + .5 * eps * n * log(.5*eps)];

    for t = 1:T
        Fcon = [Fcon, trace( Wblocks{t+1} + .5 * eps * (S \ Wblocks{t+1}) - 2*Ew(:,:,t) ) - eps / 2 * n * log(geomean(Ew(:,:,t) - eps/4*eye(n))) <= ...
                  rho_w - trace(Wh{t}) - eps/2 * (logdet(S) + logdet(Wh{t})) + .5 * eps * n * log(.5*eps)];
            
        Fcon = [Fcon, trace( Vblocks{t} + .5 * eps * (S \ Vblocks{t}) - 2*Ev(:,:,t) ) - eps / 2 * p * log(geomean(Ev(:,:,t) - eps/4*eye(p))) <= ...
                  rho_v - trace(Vh{t}) - eps/2 * (logdet(S) + logdet(Vh{t})) + .5 * eps * n * log(.5*eps)];
    end

    M = R + H'*Q*H;
    
    term1 = trace((D' * U_star' * M * U_star * D + 2 * G' * Q * H * U_star * D + G' * Q * G) * W);
    
    term2 = trace((U_star' * M * U_star) * V);
    
    obj = term1 + term2;
    
    % Solve
    options = sdpsettings('solver','mosek','verbose',0);
    sol = optimize(Fcon, -obj, options);
    if sol.problem
        disp(sol.info);
    else
        W_opt  = value(W);
        V_opt  = value(V);
    end
    cost = value(obj);
end