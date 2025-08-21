function [cost, W_opt, V_opt] = Wasserstein_SDP(T, n, m, p, rho, X0h, Wh, Vh, H, G, D, QT, RT)
    % This function computes the Wasserstein cost for the Distributionally
    % Robust LQG
    
    % Decision variables
    for t = 1:T+1
        Wblocks{t} = sdpvar(n,n);
    end
    W = blkdiag(Wblocks{:});
    for t = 1:T
        Vblocks{t} = sdpvar(p,p);
    end
    V  = blkdiag(Vblocks{:});

    M  = sdpvar(m*T, p*T,'full');
    F  = sdpvar(m*T,m*T);
    Ex0 = sdpvar(n,n);
    Ew  = sdpvar(n,n,T);
    Ev  = sdpvar(p,p,T);
    
    % Constraints
    % Positive definitiness 
    Fcon = [F >= 0, Ex0 >= 0, Wblocks{1} >= min(eig(X0h))*eye(n)];
    
    for t = 1:T
        Fcon = [Fcon, Ew(:,:,t) >= 0, Ev(:,:,t) >= 0, Vblocks{t} >= min(eig(Vh{t}))*eye(p), Wblocks{t+1} >= min(eig(Wh{t}))*eye(n)];
    end
    
    % Gelbrich bound for each covariance
    Fcon = [Fcon, trace( Wblocks{1} + X0h - 2*Ex0 ) <= rho];
    
    for t = 1:T
      Fcon = [Fcon, trace( Wblocks{t+1} + Wh{t} - 2*Ew(:,:,t) ) <= rho, ...
              trace( Vblocks{t} + Vh{t}  - 2*Ev(:,:,t) ) <= rho];
    end
    
    % Linearization using Schur's complement of the nonlinear part
    Fcon = [Fcon, [sqrtm(X0h) * Wblocks{1} * sqrtm(X0h), Ex0; Ex0, eye(n)] >= 0 ];
    
    for t=1:T
      tmp1 = sqrtm(Wh{t});
      tmp2 = sqrtm(Vh{t});
      Fcon = [Fcon, [tmp1 * Wblocks{t+1} * tmp1, Ew(:,:,t); Ew(:,:,t), eye(n)] >= 0];
      Fcon = [Fcon, [tmp2 * Vblocks{t} * tmp2, Ev(:,:,t); Ev(:,:,t), eye(p)] >= 0];
    end
    
    % Upper triangular M
    for t = 1:T
      for s = 1:t
        row_idx = (t-1)*m + (1:m);
        col_idx = (s-1)*p + (1:p);
        Fcon = [Fcon, M( row_idx, col_idx ) == 0];
      end
    end
    
    % Schur complement (A.21) 
    Z = H' * QT * G * W * D' + M/2;
    X = D * W * D' + V;
    Fcon = [Fcon, [F Z; Z' X] >= 0];
    
    % Objective
    obj = trace(G' * QT * G * W) - trace(F / (RT + H' * QT * H));
    
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