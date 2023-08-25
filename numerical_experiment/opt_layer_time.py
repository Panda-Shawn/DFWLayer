import cvxpy as cp
import time
from util import *
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpylayers.torch import CvxpyLayer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
Consider the optimization problem as follows:
    min_x 1/2 x^T@P@x + q^T@x
    s.t. sigma |w_i x_i| <= t
'''
THRES = 1e-4


def frank_wolfe(Pi, qi, wi, t):
    begin = time.time()
    n = qi.shape[0]
    xk = torch.zeros(n).to(device)

    thres = THRES
    
    res = [1000, -100]
    beta = 1e0
    time1 = time.time()
    L = power_iteration(Pi)
    tim2 = time.time()
    stop = time.time()
    count = 0
    while abs((res[-1]-res[-2])/res[-2]) > thres:
        if count%30==29: beta = beta*2
        count+=1
        dfk = (Pi @ xk + qi)
        dfk_tw = dfk * t / (wi+1e-8)
        z_hat = torch.abs(beta * dfk_tw) - torch.max(torch.abs(beta * dfk_tw))
        eik = torch.softmax(z_hat, dim=0)
        sk = -t / (wi+1e-8) * torch.sign(dfk_tw) * eik
        
        dk = xk - sk
        lnorm = L*torch.linalg.norm(dk, 2)**2
        gammak = torch.clamp(dfk@dk/(lnorm), torch.tensor(0).to(device), torch.tensor(1).to(device))
        xk = xk - gammak * dk

        re = 0.5 * (xk.T @ Pi @ xk) + qi.T @ xk
        res.append(re.detach().cpu().numpy())

    stop2 = time.time()
    xk.sum().backward()
    grad = qi.grad
    stop3 = time.time()


    print("initialization: ", stop-begin)
    print("computing L: ", tim2-time1)
    print("forward time: ", stop2-stop)
    print("backward time: ", stop3-stop2)
    print("total: ", stop3-begin)
    print("The optimal value is", (0.5 * (xk.T @ Pi @ xk) + qi.T @ xk).item())
    print("count", count)

    return (xk, grad, [stop-begin, tim2-time1, stop2-stop, stop3-stop2, stop3-begin])


def alt_diff(Pi, qi, Ai, bi, Gi, hi):
    Pi, qi, Ai, bi, Gi, hi = Pi.detach(), qi.detach(), Ai.detach(), bi.detach(), Gi.detach(), hi.detach()
    begin = time.time()
    n, m, d = qi.shape[0], bi.shape[0], hi.shape[0]
    xk = torch.zeros(n).to(device)
    sk = torch.zeros(d).to(device)
    lamb = torch.zeros(m).to(device)
    nu = torch.zeros(d).to(device)
    
    
    dxk = torch.zeros((n,n)).to(device)
    dsk = torch.zeros((d,n)).to(device)
    dlamb = torch.zeros((m,n)).to(device)
    dnu = torch.zeros((d,n)).to(device)
    
    rho = 1.0
    thres = THRES
    R = - torch.linalg.inv(Pi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
    
    res = [1000, -100]
    
    ATb = rho * Ai.T @ bi
    GTh = rho * Gi.T @ hi
    stop = time.time()
    cosins = []
    edis = []
    tmp_eye = torch.eye(n).to(device)
    tmp_ones = torch.ones((1,n)).to(device)

    while abs((res[-1]-res[-2])/res[-2]) > thres:
        xk = R @ (qi + Ai.T @ lamb + Gi.T @ nu - ATb + rho * Gi.T @ sk - GTh)
        coef1 = tmp_eye + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk
        dxk = R @ coef1
        
        sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))
        tmp_sign = sgn(sk).to(device).reshape(d,1)
        
        dsk = (-1 / rho) * tmp_sign @ tmp_ones * (dnu + rho * Gi @ dxk)

        lamb = lamb + rho * (Ai @ xk - bi)
        dlamb = dlamb + rho * (Ai @ dxk)

        nu = nu + rho * (Gi @ xk + sk - hi)
        dnu = dnu + rho * (Gi @ dxk + dsk)

        re = 0.5 * (xk.T @ Pi @ xk) + qi.T @ xk
        res.append(re.detach().cpu().numpy())
   
    stop2 = time.time()
    print("initialization: ", stop-begin)
    print("for and back: ", stop2-stop)
    print("total: ", stop2-begin)
    print("The optimal value is", res[-1])

    return (xk, dxk, [stop-begin, stop2-stop, stop2-begin])


def cvxpylayer(Pi, qi, wi, ti):
    begin = time.time()
    P_np = Pi.cpu().numpy()
    w_np = wi.cpu().numpy()
    t_np = ti.cpu().numpy()
    
    q0 = cp.Parameter(qi.shape[0])
    x = cp.Variable(qi.shape[0])
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, psd_wrap(P_np)) + q0.T @ x), [cp.norm1(cp.multiply(w_np, x)) <= t_np])
    stop = time.time()
    layer = CvxpyLayer(prob, parameters=[q0], variables=[x])
    stop2 = time.time()

    solution, = layer(qi, solver_args={'mode': 'dense', 'eps': THRES})
    stop3 = time.time()
    solution.sum().backward()
    q_g = qi.grad.cpu().numpy()
    stop4 = time.time()
    print("initialization: ", stop-begin)
    print("cananolizaition: ", stop2-stop)
    print("forward: ", stop3-stop2)
    print("backward: ", stop4-stop3)
    print("total: ", stop4-begin)
    

    x_cvx = solution.detach()
    re = (1/2) * x_cvx.T @ Pi @ x_cvx + qi.T @ x_cvx
    print("The optimal value is", re.detach().cpu().numpy())
    return (x_cvx, q_g, [stop-begin, stop2-stop, stop3-stop2, stop4-stop3, stop4-begin])

