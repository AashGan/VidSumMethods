import cupy as cnp
import numpy as np

# This should have calc scatters, one of two KTS functions

def cpd_nonlin_cupy(J,ncp,lmax=100000,backtrack = True):
    "Scatter matrix is assumed to be precomputed"
    m = int(ncp)
    lmin = 1 # Code is yet to be made for any minimum length/maximum length
    (n, n1) = J.shape
    lmin,lmax = 1,n
    I = np.full((m+1,n+1),1e9)
    I[0,lmin:lmax] = J[0,lmin-1:lmax-1]
    if backtrack:
        p = cnp.zeros((m+1,n+1),dtype=int)
    else:
        p = None
    for k in range(1,m+1):
        c = J[k:,k:] + I[k-1,k:m+1].reshape(-1,1)
        I[k,k+1:I.shape[1]] = cnp.amin(c,axis=0)
        if backtrack:
            p[k,:] = cnp.argmin(c,axis=0) + k
    
    cps = np.zeros(m, dtype=int)
    
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]
    I = I[:,n]
    I[I>1e8] = np.inf
    return cps.get(),I


def calc_scatters(K):
    """
    Calculate scatter matrix:
    scatters[i,j] = {scatter of the sequence with starting frame i and ending frame j}
    """
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    print(np.diag(K).shape)
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1) # TODO: use the fact that K - symmetric
    print(K1.shape)
    print(K2.shape)
    scatters = np.zeros((n, n))

    diagK2 = np.diag(K2)

    i = np.arange(n).reshape((-1,1))
    j = np.arange(n).reshape((1,-1))
    scatters = (K1[1:].reshape((1,-1))-K1[:-1].reshape((-1,1))
                - (diagK2[1:].reshape((1,-1)) + diagK2[:-1].reshape((-1,1)) - K2[1:,:-1].T - K2[:-1,1:]) / ((j-i+1).astype(float) + (j==i-1).astype(float)))
    scatters[j<i]=0
    #code = r"""
    #for (int i = 0; i < n; i++) {
    #    for (int j = i; j < n; j++) {
    #        scatters(i,j) = K1(j+1)-K1(i) - (K2(j+1,j+1)+K2(i,i)-K2(j+1,i)-K2(i,j+1))/(j-i+1);
    #    }
    #}
    #"""
    #weave.inline(code, ['K1','K2','scatters','n'], global_dict = \
    #    {'K1':K1, 'K2':K2, 'scatters':scatters, 'n':n}, type_converters=weave.converters.blitz)

    return scatters


def cpd_auto_cupy(K,ncp,vmax,desc_rate=1,**kwargs):
    m = ncp
    K = calc_scatters(K) 
    (_, scores) = cpd_nonlin_cupy(cnp.asarray(K), m, backtrack=False)
    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling

    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)

    costs = scores.get()/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin_cupy(cnp.asarray(K), m_best, **kwargs)
    return cps,scores2


def kts_cupy(n_frames,features,vmax=1, frame_skip = 1):
    seq_len = len(features)
    picks = np.arange(0, seq_len) * frame_skip

    # compute change points using KTS

    kernel = np.matmul(features, features.T)
    
    change_points, _ ,= cpd_auto_cupy(kernel, seq_len - 1, vmax, verbose=False)
    change_points *= frame_skip
    change_points = np.hstack((0, change_points, n_frames))
    begin_frames = change_points[:-1]
    end_frames = change_points[1:]
    change_points = np.vstack((begin_frames, end_frames - 1)).T

    n_frame_per_seg = end_frames - begin_frames
    return change_points, n_frame_per_seg, picks