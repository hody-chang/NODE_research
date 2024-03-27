function grad(param,x,y_exact,theta)
    N = param.N
    Q = param.Q
    h = 2/Q
    k = 1/N
    r = param.r

    fhat, z, dzn, dzw, dzb = forward(param,x,theta)

    a = theta.a
    c = theta.c
    w = theta.w
    b = theta.b

    Pn = zeros(Q,N)
    Pn[:,N] = dzn[:,N]
    for j = N-1:-1:1
        Pn[:,j] = Pn[:,j+1] .* dzn[:,j]
    end

    L = 0
    y_diff = y_exact[1:Q] - fhat[1:Q]

    L_main = h*sum(y_diff.^2)

    L = L + r*k*sum(w.^2+b.^2)

    L = L + r/k*sum((w[2:N]-w[1:N-1]).^2 + (b[2:N]-b[1:N-1]).^2)

    dLda = 0
    dLdc = 0
    dLdw = zeros(N)
    dLdb = zeros(N)

    

    # derivative L with a
    dLda = 2 * a * r - 2 * c * h * sum((y_diff.* Pn[:,1] .* x))

    # derivative L with c
    dLdc = 2 * c * r - 2 * h * sum(y_diff .* z[:,N+1])

    # derivative L with w
    dLdw[1] = 2*r/k*(w[1]-w[2])+ 2*r*k*w[1] - 2*c*h*sum(y_diff.*Pn[:,2].*dzw[:,1])
    dLdw[N] = 2*r/k*(w[N]-w[N-1])+ 2*r*k*w[N] - 2*c*h*sum(y_diff.*dzw[:,N])
    for j=2:N-1
        dLdw[j] = 2*r/k*(2*w[j]-w[j-1]-w[j+1])+ 2*r*k*w[j] - 2*c*h*sum(y_diff.*Pn[:,j+1].*dzw[:,j])
    end

    # derivative L with b
    dLdb[1] = 2*r/k*(b[1]-b[2])+ 2*r*k*b[1] - 2*c*h*sum(y_diff.*Pn[:,2].*dzb[:,1])
    dLdb[N] = 2*r/k*(b[N]-b[N-1])+ 2*r*k*b[N] - 2*c*h*sum(y_diff.*dzb[:,N])
    for j=2:N-1
        dLdb[j] = 2*r/k*(2*b[j]-b[j-1]-b[j+1])+ 2*r*k*b[j] - 2*c*h*sum(y_diff.*Pn[:,j+1].*dzb[:,j])
    end 

    return L, L_main, dLda, dLdc, dLdw, dLdb

end

function forward(param,x,theta)

    a = theta.a
    c = theta.c
    w = theta.w
    b = theta.b

    N = param.N
    Q = param.Q
    h = 2/Q
    k = 1/N

    z = zeros(Q,N+1)
    dzn = zeros(Q,N);
    dzw = zeros(Q,N);
    dzb = zeros(Q,N);

    z[:,1] = a*x;
    for i in 1:N
        z[:,i+1] = z[:,i] + k * tanh.(w[i] * z[:,i] .+ b[i])
        dzn[:,i] = 1 .+ k * sech.(w[i] * z[:,i] .+ b[i]).^2 * w[i]
        dzw[:,i] = k * sech.(w[i] * z[:,i] .+ b[i]).^2 .* z[:,i]
        dzb[:,i] = k * sech.(w[i] * z[:,i] .+ b[i]).^2
    end

    fhat = c*z[:,N+1]
    
    return fhat, z, dzn, dzw, dzb
end

@kwdef mutable struct param
    N::Integer = 16
    Q::Integer = 128
    r::Float64 = 0
    tol::Float64 = 1e-5
    dt::Float64 = 1e-1
    Nsteep::Integer = 10000
end

@kwdef mutable struct theta
    w::Vector{Float64}
    b::Vector{Float64}
    a::Float64
    c::Float64
end

# This is the Main function for the model to Train
# Input: N, r, First w, First b
# Output: The best w, The best b and the loss

function Model_run(N,r,w_initial,b_initial)
    param_node = param()
    param_node.r = r
    param_node.N = N
    theta_node = theta(w_initial,b_initial,1,1)
    h = 2/param_node.Q
    k = 1/param_node.N
    x = (-1 .+ (1:param_node.Q) .* h .- h / 2)'
    y_exact = x + (1 .-x.^2)./3
    #y_exact = x + (1 .+ x.^2)./3
    L, L_main, dLda, dLdc, dLdw, dLdb = grad(param_node,x,y_exact,theta_node)
    tol = param_node.tol
    resid = 2*tol
    while resid > tol
        for j=1:param_node.Nsteep
            L, L_main, dLda, dLdc, dLdw, dLdb = grad(param_node,x,y_exact,theta_node)

            theta_node.w = theta_node.w - param_node.dt * dLdw
            theta_node.b = theta_node.b - param_node.dt * dLdb
        end
        resid = maximum(abs.([dLdw; dLdb]))
    end
    println(" Reg = $r "," Resid = $resid ", " Loss = $L_main ")
    return theta_node.w, theta_node.b, L_main
end