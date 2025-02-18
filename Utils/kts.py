import numpy as np
# Sources: https://github.com/TatsuyaShirakawa/KTS , https://github.com/li-plus/DSNet/blob/master/src/make_shots.py
def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Main interface

    Detect change points automatically selecting their number
        K       - kernel between each pair of frames in video
        ncp     - maximum ncp
        vmax    - special parameter
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        desc_rate - rate of descriptor sampling (vmax always corresponds to 1x)

    Note:
        - cps are always calculated in subsampled coordinates irrespective to
            desc_rate
        - lmin and m should be in agreement
    ---
    Returns: (cps, costs)
        cps   - best selected change-points
        costs - costs for 0,1,2,...,m change-points

    Memory requirement: ~ (3*N*N + N*ncp)*4 bytes ~= 16 * N^2 bytes
    That is 1,6 Gb for the N=10000.
    """
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)

    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling

    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)

    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)

    return (cps, scores2)


#from scipy import weave

def calc_scatters(K):
    """
    Calculate scatter matrix:
    scatters[i,j] = {scatter of the sequence with starting frame i and ending frame j}
    """
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1) # TODO: use the fact that K - symmetric

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

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True,
    out_scatters=None):
    """ Change point detection with dynamic programming
    K - square kernel matrix
    ncp - number of change points to detect (ncp >= 0)
    lmin - minimal length of a segment
    lmax - maximal length of a segment
    backtrack - when False - only evaluate objective scores (to save memory)

    Returns: (cps, obj)
        cps - detected array of change points: mean is thought to be constant on [ cps[i], cps[i+1] )
        obj_vals - values of the objective function for 0..m changepoints

    """
    m = int(ncp)  # prevent numpy.int64

    (n, n1) = K.shape
    assert(n == n1), "Kernel matrix awaited."

    assert(n >= (m + 1)*lmin)
    assert(n <= (m + 1)*lmax)
    assert(lmax >= lmin >= 1)

    if verbose:
        #print "n =", n
        print ("Precomputing scatters...")
    J = calc_scatters(K)

    if out_scatters != None:
        out_scatters[0] = J

    if verbose:
        print ("Inferring best change points...")
    # I[k, l] - value of the objective for k change-points and l first frames
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]

    if backtrack:
        # p[k, l] --- "previous change" --- best t[k] when t[k+1] equals l
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1,1), dtype=int)

    for k in range(1,m+1):
        for l in range((k+1)*lmin, n+1):
            tmin = max(k*lmin, l-lmax)
            tmax = l-lmin+1
            c = J[tmin:tmax,l-1].reshape(-1) + I[k-1, tmin:tmax].reshape(-1)
            I[k,l] = np.min(c)
            if backtrack:
                p[k,l] = np.argmin(c)+tmin

    #code = r"""
    ##define max(x,y) ((x)>(y)?(x):(y))
    #for (int k=1; k<m+1; k++) {
    #    for (int l=(k+1)*lmin; l<n+1; l++) {
    #        I(k, l) = 1e100; //nearly infinity
    #        for (int t=max(k*lmin,l-lmax); t<l-lmin+1; t++) {
    #            double c = I(k-1, t) + J(t, l-1);
    #            if (c < I(k, l)) {
    #                I(k, l) = c;
    #                if (backtrack == 1) {
    #                    p(k, l) = t;
    #                }
    #            }
    #        }
    #    }
    #}
    #"""

    #weave.inline(code, ['m','n','p','I', 'J', 'lmin', 'lmax', 'backtrack'], \
    #    global_dict={'m':m, 'n':n, 'p':p, 'I':I, 'J':J, \
    #    'lmin':lmin, 'lmax':lmax, 'backtrack': int(1) if backtrack else int(0)},
    #    type_converters=weave.converters.blitz)

    # Collect change points
    cps = np.zeros(m, dtype=int)

    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores


def kts(n_frames,features,vmax=1, frame_skip = 1):
      """ Receives the frame features from the CNN to do the Shot division based on KTS
      """
      n_frames = len(features) * frame_skip -1
      seq_len = len(features)
      picks = np.arange(0, seq_len) * frame_skip

      # compute change points using KTS
      kernel = np.matmul(features, features.T)
      change_points, _ = cpd_auto(kernel, seq_len - 1, vmax, verbose=False)
      change_points *= frame_skip
      change_points = np.hstack((0, change_points, n_frames))
      begin_frames = change_points[:-1]
      end_frames = change_points[1:]
      change_points = np.vstack((begin_frames, end_frames - 1)).T

      n_frame_per_seg = end_frames - begin_frames
      return change_points, n_frame_per_seg, picks