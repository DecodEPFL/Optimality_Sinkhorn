function [cost, Ws, Vs] = Sinkhorn_conic(T, n, m, p, rho_0, rho_w, rho_v, eps, X0h, Wh, Vh, H, G, D, QT, RT, S)
    % Decision variables
    for t = 1:T+1
        Wblocks{t} = sdpvar(n,n);
    end

    for t = 1:T
        Vblocks{t} = sdpvar(p,p);
    end

    M  = sdpvar(m*T, p*T,'full');
    F  = sdpvar(m*T,m*T);
    Ex0 = sdpvar(n,n);
    Ew  = sdpvar(n,n,T);
    Ev  = sdpvar(p,p,T);

    % Constraints
    % Positive definitiness 
    Fcon = [F >= 0, Ex0 >= 0, Wblocks{1} >= 0];
    
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
    
    % Upper triangular M
    for t = 1:T
      for s = 1:t
        row_idx = (t-1)*m + (1:m);
        col_idx = (s-1)*p + (1:p);
        Fcon = [Fcon, M( row_idx, col_idx ) == 0];
      end
    end
    
    Fcon = [Fcon, trace( Wblocks{1} + .5 * eps * (S \ Wblocks{1}) - 2*Ex0 ) - eps / 2 * n * log(geomean(Ex0 - eps/4*eye(n))) <= ...
                  rho_0 - trace(X0h) - eps/2 * (logdet(S) + logdet(X0h)) + .5 * eps * n * log(.5*eps)];

     % Fcon = [Fcon, trace( Wblocks{1} + .5 * eps * (S \ Wblocks{1}) - 2*Ex0 ) - eps / 2 * n * log(geomean(Ex0 - eps/4*eye(n))) <= ...
     %              rho_0 - trace(X0h) - eps/2 * (logdet(S) + logdet(X0h)) + .5 * eps * n * log(.5*eps)];

    for t = 1:T
        Fcon = [Fcon, trace( Wblocks{t+1} + .5 * eps * (S \ Wblocks{t+1}) - 2*Ew(:,:,t) ) - eps / 2 * n * log(geomean(Ew(:,:,t) - eps/4*eye(n))) <= ...
                  rho_w - trace(Wh{t}) - eps/2 * (logdet(S) + logdet(Wh{t})) + .5 * eps * n * log(.5*eps)];
            
        Fcon = [Fcon, trace( Vblocks{t} + .5 * eps * (S \ Vblocks{t}) - 2*Ev(:,:,t) ) - eps / 2 * p * log(geomean(Ev(:,:,t) - eps/4*eye(p))) <= ...
                  rho_v - trace(Vh{t}) - eps/2 * (logdet(S) + logdet(Vh{t})) + .5 * eps * n * log(.5*eps)];
    end

    W = blkdiag(Wblocks{:});
    V  = blkdiag(Vblocks{:});

    % Schur complement (A.21) 
    Z = H' * QT * G * W * D' + M/2;
    X = D * W * D' + V;
    Fcon = [Fcon, [F Z; Z' X] >= 0];

    % Objective
    obj = trace(G' * QT * G * W) - trace(F / (RT + H' * QT * H));

    fprintf('=====================================\n')
    fprintf("Solving the optimization problem...\n")
    fprintf('=====================================\n')
    options = sdpsettings('solver','mosek','verbose',0);
    sol = optimize(Fcon, -obj, options);
    cost = value(obj);
    Ws = value(W);
    Vs = value(V);

    if ~(sol.problem == 0)
        if sol.problem == 1
            cost = inf;
            return
        elseif sol.problem == 4
            cost = value(obj);
        else
            string = yalmiperror(sol.problem);
            error(string);
        end
    end
end