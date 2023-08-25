import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)
import torch
import cvxpy as cp
from torch.autograd import Function
import numpy as np
from util import *
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpylayers.torch import CvxpyLayer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
'''
Consider the optimization problem as follows:
    min_x 1/2 x^T@P@x + q^T@x
    s.t. sigma |w_i x_i| <= t
'''
THRES = 1e-3


def frank_wolfe(Pi, qi, wi, t, thres=THRES):
    n = qi.shape[0]
    xk = torch.zeros(n).to(device)
    # L = torch.max(torch.abs(torch.linalg.svd(Pi)[1]))
    L = power_iteration(Pi)
    res = [1000, -100]
    beta = 1e-1
    count = 0
    while abs((res[-1]-res[-2])/res[-2]) > thres and count < 10:
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

    return xk


def alt_diff(Pi, qi, Ai, bi, Gi, hi, thres=THRES):
    Pi, qi, Ai, bi, Gi, hi = Pi.detach(), qi.detach(), Ai.detach(), bi.detach(), Gi.detach(), hi.detach()
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
    R = - torch.linalg.inv(Pi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
    
    res = [1000, -100]
    
    ATb = rho * Ai.T @ bi
    GTh = rho * Gi.T @ hi
    count = 0
    while abs((res[-1]-res[-2])/res[-2]) > thres and count < 10:
        count+=1
        xk = R @ (qi + Ai.T @ lamb + Gi.T @ nu - ATb + rho * Gi.T @ sk - GTh)
        coef1 = torch.eye(n).to(device) + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk
        dxk = R @ coef1
        
        sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))
        dsk = (-1 / rho) * sgn(sk).to(device).reshape(d,1) @ torch.ones((1,n)).to(device) * (dnu + rho * Gi @ dxk)

        lamb = lamb + rho * (Ai @ xk - bi)
        dlamb = dlamb + rho * (Ai @ dxk)

        nu = nu + rho * (Gi @ xk + sk - hi)
        dnu = dnu + rho * (Gi @ dxk + dsk)

        re = 0.5 * (xk.T @ Pi @ xk) + qi.T @ xk
        res.append(re.detach().cpu().numpy())
    return (xk, dxk)


def altdiff_layer(eps=1e-3):
    class Newlayer(Function):
        @staticmethod
        def forward(ctx, P_, q_, w_, t_):
            n = q_.shape[1]
            P = P_.cpu().numpy()
            q = q_.cpu().numpy()
            w = w_.cpu().numpy()
            t = t_.cpu().numpy()
            # Define and solve the CVXPY problem.
            optimal = []
            gradient = []

            for i in range(len(P)):
                Pi, qi, wi, ti = P[i], q[i], w[i], t[i]

                tmp_small = np.zeros(n) + 1e-8
                P_aug = np.zeros((2*n, 2*n))
                P_aug[:n, :n] = Pi
                P_aug[n:, n:] = np.diag(tmp_small)

                q_aug = np.zeros(2*n)
                q_aug[:n] = qi
                q_aug[n:] = tmp_small
                Ai = np.zeros((1, 2*n))
                bi = np.zeros((1))
                tmp = np.ones(n)
                G_1 = np.concatenate([-np.diag(wi+1e-8), -np.diag(tmp)], axis=1)
                G_2 = np.concatenate([np.diag(wi+1e-8), -np.diag(tmp)], axis=1)
                G_3 = np.concatenate([np.zeros((n, n)), -np.diag(tmp)], axis=1)
                G_4 = np.concatenate([np.zeros((1, n)), np.ones((1, n))], axis=1)
                Gi = np.concatenate([G_1, G_2, G_3, G_4], axis=0)
                hi = np.zeros((3*n+1))
                hi[-1] = ti

                P_th = torch.from_numpy(P_aug).float().to(P_.device)
                q_th = torch.from_numpy(q_aug).float().to(P_.device)
                A_th = torch.from_numpy(Ai).float().to(P_.device)
                b_th = torch.from_numpy(bi).float().to(P_.device)
                G_th = torch.from_numpy(Gi).float().to(P_.device)
                h_th = torch.from_numpy(hi).float().to(P_.device)

                xk, dxk = alt_diff(P_th, q_th, A_th, b_th, G_th, h_th, thres=eps)

                optimal.append(xk[:n])
                gradient.append(dxk[:n, :n])

            ctx.save_for_backward(torch.stack(gradient))
            return torch.stack(optimal)

        @staticmethod
        def backward(ctx, grad_output):
            # only call parameters q
            grad = ctx.saved_tensors

            grad_alls = []
            for i in range(len(grad[0])):
                grad_all = grad_output[i] @ grad[0][i]
                grad_alls.append(grad_all)
            return (None, torch.stack(grad_alls), None, None)

    return Newlayer.apply


def dfw_layer(eps=1e-4):
    def batch_dfw(P, q, w, t):
        optimal = []
        for i in range(q.shape[0]):
            # print(i)
            Pi, qi, wi, ti = P[i], q[i], w[i], t[i]
            xk = frank_wolfe(Pi, qi, wi, ti, thres=eps)
            optimal.append(xk)
        return torch.stack(optimal)

    return batch_dfw


def cvxpy_layer(eps=1e-4):
    def batch_cvxlayer(P, q, w, t):
        P_np = P.cpu().numpy()
        w_np = w.cpu().numpy()
        t_np = t.cpu().numpy()
        optimal = []
        for i in range(q.shape[0]):
            Pi, qi, wi, ti = P_np[i], q[i], w_np[i], t_np[i]
            q0 = cp.Parameter(qi.shape[0])
            x = cp.Variable(qi.shape[0])
            prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, psd_wrap(Pi)) + q0.T @ x), [cp.norm1(cp.multiply(wi, x)) <= ti])
            layer = CvxpyLayer(prob, parameters=[q0], variables=[x])
            solution, = layer(qi, solver_args={'mode': 'dense', 'eps': eps})
            optimal.append(solution)
        return torch.stack(optimal)

    return batch_cvxlayer

