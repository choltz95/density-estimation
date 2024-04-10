import sys
sys.path.insert(0, '../pyLDLE2/')

import numpy as np
from pyLDLE2 import util_, visualize_, datasets
from scipy.sparse import coo_matrix
from scipy import optimize
from scipy.special import erf, erfinv
from matplotlib import pyplot as plt
from scipy.stats import chi2, spearmanr
from sklearn.decomposition import PCA
from scipy.linalg import svd

default_opts = {
    'd': 2,
    'k_nn': 128,
    'k_tune': 128,
    'ds': False,
    'p': 0.99,
    'h': 0,
    'ds': False,
    's': 0.3,
    'local_pca': False,
    'newton_maxiter': 1000,
    'H': 0
}

def sinkhorn(K, maxiter=10000, delta=1e-20, eps=0):
    """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
    D = np.array(K.sum(axis=1)).squeeze()
    d0 = 1./(D+eps)
    d1 = 1./(K.dot(d0)+eps)
    d2 = 1./(K.dot(d1)+eps)
    for tau in range(maxiter):
        if np.max(np.abs(d0 / d2 - 1)) < delta:
            print('Sinkhorn converged at iter:', tau)
            break
        d3 = 1. / (K.dot(d2) + eps)
        d0=d1.copy()
        d1=d2.copy()
        d2=d3.copy()
    d = np.sqrt(d2 * d1)
    K.data = K.data*d[K.row]*d[K.col]
    return K, d

def compute_m0(bx, h, d=2):
    return 0.5*(np.pi**(d/2))*(1+erf(bx/h))

def compute_m1(bx, h, d=2):
    return -0.5*(np.pi**((d-1)/2))*np.exp(-(bx/h)**2)

def compute_m(bx, h):
    return compute_m1(bx,h)/compute_m0(bx,h)

def compute_A(bx, h, d=2):
    m1 = compute_m1(bx, h, d) 
    m0 = compute_m0(bx, h, d) 
    return 2*m1*(m1*(1-2*bx/h) + np.pi**(d/2) - m0)

def compute_B(bx, h, d=2, H=0):
    m1 = compute_m1(bx, h, d) 
    return np.pi**(d) + m1*(0.5*h*(d-1)*H - 2*(np.pi**(d/2))*bx/h)

def compute_C(bx, h, d=2, H=0):
    A = compute_A(bx, h, d)
    B = compute_B(bx, h, d, H)
    return (B-np.sqrt(B**2 -4*(np.pi**d)*A), 2*A)
    
def epanechnikov_kernel(dist, eps):
    return (1-dist**2/eps)*(dist > 0)*(dist < np.sqrt(eps))

def estimate_bx(X, opts=default_opts, ret_K_D=False):
    d = opts['d']
    h = opts['h']
    ds = opts['ds']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]

    # set h
    if h <= 0:
        h_cand = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
        h = np.min(h_cand)

    print('h:', h, flush=True)
    
    if ('K' in opts) and (opts['K'] is not None):
        K = opts['K']
        D = opts['D']
    else:
        # standard kernel
        K = np.exp(-neigh_dist**2/h**2)
        n = X.shape[0]
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))

        # symmetrize the kernel
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data

        # if doubly stochastic
        if ds:
            K = K.tocoo()
            K, D = sinkhorn(K)
            K = K.tocsr()
        else:
            D = None
    
    # Compute ||mu_i||
    
    if ds:
        s = opts['s']
    
    mu_hN_norm = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        temp = X-X[i,:][None,:]
        mu_hN_norm[i] = np.linalg.norm(K.getrow(i).dot(temp))

    eps_pca = np.max(neigh_dist)**2
    mu_norm = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if opts['local_pca']:
#             n_i = neigh_ind[i]
#             X_i = X[n_i,:].T # p x N_i
#             X_i = X_i - X[i,:][:,None]
#             X_i_norm = np.linalg.norm(X_i, axis=0) # N_i dimensional
#             D_i = np.sqrt(epanechnikov_kernel(X_i_norm, eps_pca)) #N_i dimensional
#             B_i = X_i * D_i[None,:]
#             U_i, Sigma_i, V_iT = svd(B_i)
#             temp = Sigma_i[:d][:,None]*V_iT[:d,:]
#             temp = temp.T
            X_i_nbrs = X[neigh_ind[i,:].tolist()+[i],:]
            pca = PCA(n_components=d)
            y = pca.fit_transform(X_i_nbrs)
            X_rec = pca.inverse_transform(y)
            temp = X_rec[:-1,:] - X_rec[-1,:][None,:]
        else:
            temp = X[neigh_ind[i,:],:]-X[i,:][None,:]
            
        if ds:
            mu_norm[i] = np.linalg.norm(K.getrow(i).power(s)[0,neigh_ind[i,:]].dot(temp))
        else:
            mu_norm[i] = np.linalg.norm(K.getrow(i)[0,neigh_ind[i,:]].dot(temp))
            
    if ds:
        c_num = h*(np.array((K.power(s)).sum(axis=1)).flatten())/(np.sqrt(np.pi)*np.sqrt(s))
    else:
        c_num = h*np.array(K.sum(axis=1)).flatten()/np.sqrt(np.pi)

    c_denom = mu_norm
    #c = c_num/(c_denom+1e-20)

    if ds:
        def F(x):
            return c_denom*((1+erf(np.sqrt(s)*x/h)))*np.exp(s*(x**2/h**2))-c_num
        def F_prime(x):
            return (c_denom/h)*(2*np.sqrt(s)/np.sqrt(np.pi) + 2*s*(1+erf(np.sqrt(s)*x/h))*np.exp(s*x**2/h**2)*x/h)
    else:
        def F(x):
            return c_denom*(1+erf(x/h))*np.exp(x**2/h**2)-c_num
        def F_prime(x):
            return (c_denom/h)*(2/np.sqrt(np.pi) + 2*(1+erf(x/h))*np.exp(x**2/h**2)*x/h)

    if ds:
        bx_init = h*np.sqrt(np.maximum(0, (-np.log(2*c_denom+1e-30)+np.log(c_num+1e-30))/s))
    else:
        bx_init = h*np.sqrt(np.maximum(0, -np.log(2*c_denom+1e-30)+np.log(c_num+1e-30)))
        
    bx = optimize.newton(F, bx_init, F_prime, maxiter=opts['newton_maxiter'])
    bx = np.maximum(bx, 0)
    
    if ret_K_D:
        return bx, bx_init, K, D
    return bx, bx_init

def compute_SRCC(x, y, prctile=10):
    x = x.copy()
    y = y.copy()
    x_ = np.percentile(x, prctile)
    y_ = np.percentile(y, prctile)
    x = x[x < x_]
    y = y[y < y_]
    
    assert len(x) == len(y)
    return spearmanr(x, y)


def estimate_q(X, opts=default_opts, bx=None):
    ds = opts['ds']
    s = opts['s']
    h = opts['h']
    d = opts['d']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]
    
    # set h
    if h <= 0:
        h_cand = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
        h = np.min(h_cand)
        
    print('h:', h, flush=True)
    
    # standard kernel
    K = np.exp(-neigh_dist**2/h**2)
    n = X.shape[0]
    source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
    K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))

    # symmetrize the kernel
    K = K + K.T
    ones_K_like = ones_K_like + ones_K_like.T
    K.data /= ones_K_like.data
    
    K_old = K.copy()
    
    # if doubly stochastic
    if ds:
        K = K.tocoo()
        K, D = sinkhorn(K)
        K = K.tocsr()
        print('s:', s, flush=True)
        
    if bx is not None:
        if ds:
            if opts['q_est_type'] == 1:
                m0_h = compute_m0(bx, h)
                m0_h_s = compute_m0(bx, h/np.sqrt(s))
                m0 = ((m0_h**(-s))*m0_h_s)
            elif opts['q_est_type'] == 2:
                m0_h = compute_m0(bx, h/np.sqrt(s))
            else:
                m0_h = compute_m0(bx, h)
        else:
            m0 = compute_m0(bx, h)
    else:
        m0 = 1
    
    n = X.shape[0]
    if ds:
        if opts['q_est_type'] == 1:
            Z = ((n-1)**s)*(h**(d*(s-1)))*(s**(d/2))
            #Z = (((m0_h**(d/2))*(n-1))**s)*(h**(d*(s-1)))*(s**(d/2))
            f = (np.array((K.power(s)).sum(axis=1)).flatten()/(n-1))*Z/m0
            q = f**(1/(1-s))
        elif opts['q_est_type'] == 1:
            f = (s**(d/2)) * np.array((K.power(s)).sum(axis=1)).flatten()/(D.power(2*s))
            q = f/((np.pi**(d*s*0.5)) * (h**d) * m0_h)
        else:
            rho = D*np.sqrt((n-1)*((np.pi*h**2)**(d/2)))
            q = (rho**2)*(np.pi**(d/2))*m0_h
    else:
        q = np.array(K_old.sum(axis=1)).flatten()/((n-1)*h**d)
        q = q/m0
        
    return q

def estimate_bx_new(X, opts=default_opts, ret_K_D=False):
    d = opts['d']
    h = opts['h']
    H = opts['H']
    
    def compute_D(h_scale):
        h = opts['h']
        # compute nearest neighbors
        neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
        neigh_dist = neigh_dist[:,1:]
        neigh_ind = neigh_ind[:,1:]

        # set h
        if h <= 0:
            h_cand = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
            h = np.min(h_cand)

        h = h*h_scale
        print('h:', h, flush=True)

        # standard kernel
        K = np.exp(-neigh_dist**2/h**2)
        n = X.shape[0]
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))

        # symmetrize the kernel
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data
        
        K = K.tocoo()
        K, D = sinkhorn(K)
        return D, h
    
    D_h, h = compute_D(1)
    D_hb2, hb2 = compute_D(0.5)
    
    D_h = (h**(d/2)) * D_h
    D_hb2 = (hb2**(d/2)) * D_hb2
    
    def F(bx):
        C_hb2_num, C_hb2_denom = compute_C(bx, hb2, d, H)
        C_h_num, C_h_denom = compute_C(bx, h, d, H)
        return D_h*C_hb2_num*C_h_denom - D_hb2*C_h_num*C_hb2_denom
        
    np.random.seed(42)
    bx_init = 0.01 * np.random.uniform(0, 1, D_h.shape[0])
    bx = optimize.newton(F, bx_init, maxiter=opts['newton_maxiter'])
    bx = np.maximum(bx, 0)
    return bx

def estimate_Hx(X, bx, opts=default_opts, ret_K_D=False):
    d = opts['d']
    h = opts['h']

    def compute_D(h_scale):
        h = opts['h']
        # compute nearest neighbors
        neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
        neigh_dist = neigh_dist[:,1:]
        neigh_ind = neigh_ind[:,1:]

        # set h
        if h <= 0:
            h_cand = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
            h = np.min(h_cand)

        h = h*h_scale
        print('h:', h, flush=True)

        # standard kernel
        K = np.exp(-neigh_dist**2/h**2)
        n = X.shape[0]
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))

        # symmetrize the kernel
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data
        
        K = K.tocoo()
        K, D = sinkhorn(K)
        return D, h
    
    D_h, h = compute_D(1)
    D_hb2, hb2 = compute_D(0.5)
    
    D_h = (h**(d/2)) * D_h
    D_hb2 = (hb2**(d/2)) * D_hb2
    
    def F(H):
        C_hb2_num, C_hb2_denom = compute_C(bx, hb2, d, H)
        C_h_num, C_h_denom = compute_C(bx, h, d, H)
        return D_h*C_hb2_num*C_h_denom - D_hb2*C_h_num*C_hb2_denom
        
    H_init = np.zeros(D_h.shape[0])
    H = optimize.newton(F, H_init, maxiter=opts['newton_maxiter'])
    return H