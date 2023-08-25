import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)
from opt_layer_time import alt_diff, cvxpylayer, frank_wolfe
from util import *
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def param_random_for_norms(n, seed):
    np.random.seed(seed)
    t = torch.abs(torch.from_numpy(np.random.randn(1))).to(device)
    P = np.random.randn(n, n)

    P_np = P.T @ P + n * np.eye(n)
    P = torch.from_numpy(P_np).to(device)

    q_np =np.random.randn(n)
    q = torch.from_numpy(q_np).to(device)
    
    w_np = np.random.uniform(0.5, 1.5, n)
    w = torch.from_numpy(w_np).to(device)

    return P.float(), q.float().requires_grad_(), w.float(), t.float()


def param_random_for_qp(n, seed):
    np.random.seed(seed)
    t_np = np.random.randn(1)
    t = torch.abs(torch.from_numpy(t_np)).to(device)

    P = np.random.randn(n, n)
    P_np = P.T @ P + n * np.eye(n)
    P = torch.from_numpy(P_np).to(device)
    tmp_small = torch.zeros(n) + 1e-8
    P_aug = torch.zeros(2*n, 2*n).to(device)
    P_aug[:n, :n] = P
    P_aug[n:, n:] = torch.diag(tmp_small)

    q = torch.from_numpy(np.random.randn(n)).to(device)
    q_aug = torch.zeros(2*n).to(device)
    q_aug[:n] = q
    q_aug[n:] = tmp_small
    q_aug = q_aug

    w_np = np.random.uniform(0.5, 1.5, n)
    w = torch.from_numpy(w_np)
    
    A = torch.zeros((1, 2*n)).to(device)
    b = torch.zeros((1)).to(device)

    tmp = np.ones(n)
    G_1 = np.concatenate([-np.diag(w_np+1e-8), -np.diag(tmp)], axis=1)
    G_2 = np.concatenate([np.diag(w_np+1e-8), -np.diag(tmp)], axis=1)
    G_3 = np.concatenate([np.zeros((n, n)), -np.diag(tmp)], axis=1)
    G_4 = np.concatenate([np.zeros((1, n)), np.ones((1, n))], axis=1)
    G_np = np.concatenate([G_1, G_2, G_3, G_4], axis=0)
    G = torch.from_numpy(G_np).to(device)
    h_np = np.zeros((3*n+1))
    h_np[-1] = t_np
    h = torch.from_numpy(h_np).to(device)


    return P_aug.float(), q_aug.float().requires_grad_(), A.float(), b.float(), G.float(), h.float()


def test_QP_Frank_Wolfe(P, q, w, t):
    for i in range(1):
        begin_alt = time.time()
        print(f'(trial {i})')
        xk, dxk, time_list = frank_wolfe(P, q, w, t)
        end_alt = time.time()
        return xk, dxk, end_alt - begin_alt, time_list


def test_QP_Alt_Diff(P, q, A, b, G, h):
    for i in range(1):
        begin_alt = time.time()
        print(f'(trial {i})')
        xk, dxk, time_list = alt_diff(P, q, A, b, G, h)
        b_f = torch.sum(dxk[:n, :n], axis=0)
        end_alt = time.time()
        return xk[:n], b_f, end_alt - begin_alt, time_list


def test_QP_cvxpylayers(P, q, w, t):
    for i in range(1):
        begin_cvx = time.time()
        print(f'(trial {i})')
        x_cvx, b_g, time_list = cvxpylayer(P, q, w, t)
        end_cvx = time.time()
        return x_cvx, b_g, end_cvx - begin_cvx, time_list


if __name__ == '__main__':
    n = 300
    seeds = [100, 200, 300, 400, 500]
    time_cvx = []
    time_ad = []
    time_fw = []
    cos_ad = []
    cos_fw = []
    edis_ad = []
    edis_fw = []
    cost_cvx = []
    cost_ad = []
    cost_fw = []
    cos_ad_tau = []
    cos_fw_tau = []
    edis_ad_tau = []
    edis_fw_tau = []
    step = 1000
    for i in range(5):
        seed = seeds[i]

        P, q, w, t = param_random_for_norms(n, seed)
        x_cvx, q_g, cvx, time_list = test_QP_cvxpylayers(P, q, w, t)
        violation = torch.relu(torch.norm(w*x_cvx, p=1)-t)
        cost_cvx.append(violation.detach().cpu().numpy())
        time_cvx.append(time_list)

        P, q, w, t = param_random_for_norms(n, seed)
        x_fw, q_fw, fw, time_list = test_QP_Frank_Wolfe(P, q, w, t)
        violation = torch.relu(torch.norm(w*x_fw, p=1)-t)
        time_fw.append(time_list)

        P, q, A, b, G, h = param_random_for_qp(n, seed)
        x_ad, q_ad, Alt_Diff, time_list = test_QP_Alt_Diff(P, q, A, b, G, h)
        violation = torch.relu(torch.norm(w*x_ad, p=1)-t)
        cost_ad.append(violation.detach().cpu().numpy())
        time_ad.append(time_list)

        cos_fw.append(cosDis(q_g, q_fw.detach().cpu().numpy()))
        cos_ad.append(cosDis(q_g, q_ad.detach().cpu().numpy()))
        edis_fw.append(np.sqrt(np.sum(np.square(x_cvx.detach().cpu().numpy()-x_fw.detach().cpu().numpy()))))
        edis_ad.append(np.sqrt(np.sum(np.square(x_cvx.detach().cpu().numpy()-x_ad.detach().cpu().numpy()))))

    print("time for cvx:", np.mean(time_cvx, axis=0), np.std(time_cvx, axis=0))
    print("time for fw:", np.mean(time_fw, axis=0), np.std(time_fw, axis=0))
    print("time for ad:", np.mean(time_ad, axis=0), np.std(time_ad, axis=0))

    print("Cosine distance between Frank-Wolfe and cvxpy is ", np.mean(cos_fw), np.std(cos_fw))
    print("Solution Euclidean distance between Frank-Wolfe and cvxpy is ", np.mean(edis_fw), np.std(edis_fw))
    print("Cosine distance between Alt-Diff and cvxpy is ", np.mean(cos_ad), np.std(cos_ad))
    print("Solution Euclidean distance between Alt-Diff and cvxpy is ", np.mean(edis_ad), np.std(edis_ad))

    print("cost for cvx:", np.max(cost_cvx), np.std(cost_cvx), np.mean(cost_cvx))
    print("cost for fw:", np.max(cost_fw), np.std(cost_fw), np.mean(cost_fw))
    print("cost for ad:", np.max(cos_ad), np.std(cost_ad), np.mean(cost_ad))

