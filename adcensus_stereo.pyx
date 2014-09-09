#cython: boundscheck=False
#cython: wraparound=False

cdef extern from "math.h":
    float INFINITY

import sys
from scipy.ndimage.filters import gaussian_filter, sobel
import numpy as np
cimport numpy as np

cdef int height, width, disp_scale
cdef int disp_max = 64

cdef int L1 = 1
cdef int L2 = 3
cdef int tau1 = 20
cdef int tau2 = 6

cdef double pi1 = 1.
cdef double pi2 = 3.
cdef int tau_so = 15

cdef int tau_s = 20
cdef double tau_h = 0.4

cdef int tau_e = 10 


def init(int h, int w, int d, int scale):
    global height, width, disp_max, disp_scale

    height = h
    width = w
    disp_max = d
    disp_scale = scale

def ad_vol(np.ndarray[np.float32_t, ndim=2] x0, np.ndarray[np.float32_t, ndim=2] x1):
    cdef np.ndarray[np.float64_t, ndim=3] res
    cdef int d, i, j, c

    H = x0.shape[0]
    W = x0.shape[1]
    print H, W
    res = np.zeros((H, W, disp_max))
    for d in range(disp_max):
        for i in range(H):
            for j in range(W):
                if j - d < 0:
                    res[i,j,d] = INFINITY
                else:
                    res[i,j,d] += abs(x0[i,j] - x1[i,j-d])
                    res[i,j,d] /= 3
    return res

# def census_transform(np.ndarray[np.float64_t, ndim=3] x):
#     cdef np.ndarray[np.int_t, ndim=3] cen
#     cdef int i, j, ii, jj, k, ind, ne

#     ne = np.random.randint(2**31)
#     cen = np.zeros((height, width, 63 * 3), dtype=np.int)
#     for i in range(height):
#         for j in range(width):
#             ind = 0
#             for ii in range(i - 3, i + 4):
#                 for jj in range(j - 4, j + 5):
#                     for k in range(3):
#                         if 0 <= ii < height and 0 <= jj < width:
#                             cen[i, j, ind] = x[ii, jj, k] < x[i, j, k]
#                         else:
#                             cen[i, j, ind] = ne
#                         ind += 1
#     return cen

# cdef int cross_coditions(int i, int j, int ii, int jj, int iii, int jjj,
#                          np.ndarray[np.float64_t, ndim=3] x):
#     cdef double v0, v1, v2

#     if not (0 <= ii < height and 0 <= jj < width): return 0

#     if abs(i - ii) == 1 or abs(j - jj) == 1: return 1

#     # rule 1
#     if abs(x[i,j,0] - x[ii,jj,0]) >= tau1: return 0
#     if abs(x[i,j,1] - x[ii,jj,1]) >= tau1: return 0
#     if abs(x[i,j,2] - x[ii,jj,2]) >= tau1: return 0

#     if abs(x[ii,jj,0] - x[iii,jjj,0]) >= tau1: return 0
#     if abs(x[ii,jj,1] - x[iii,jjj,1]) >= tau1: return 0
#     if abs(x[ii,jj,2] - x[iii,jjj,2]) >= tau1: return 0

#     # rule 2
#     if abs(i - ii) >= L1 or abs(j - jj) >= L1: return 0

#     # rule 3
#     if abs(i - ii) >= L2 or abs(j - jj) >= L2:
#         if abs(x[i,j,0] - x[ii,jj,0]) >= tau2: return 0
#         if abs(x[i,j,1] - x[ii,jj,1]) >= tau2: return 0
#         if abs(x[i,j,2] - x[ii,jj,2]) >= tau2: return 0
        
#     return 1
    

# def cross(np.ndarray[np.float64_t, ndim=3] x):
#     cdef np.ndarray[np.int_t, ndim=3] res
#     cdef int i, j, yn, ys, xe, xw
    
#     res = np.empty((height, width, 4), dtype=np.int)
#     for i in range(height):
#         for j in range(width):
#             res[i,j,0] = i - 1
#             res[i,j,1] = i + 1
#             res[i,j,2] = j - 1
#             res[i,j,3] = j + 1
#             while cross_coditions(i,j,res[i,j,0],j,res[i,j,0]+1,j,x): res[i,j,0] -= 1
#             while cross_coditions(i,j,res[i,j,1],j,res[i,j,1]-1,j,x): res[i,j,1] += 1
#             while cross_coditions(i,j,i,res[i,j,2],i,res[i,j,2]+1,x): res[i,j,2] -= 1
#             while cross_coditions(i,j,i,res[i,j,3],i,res[i,j,3]-1,x): res[i,j,3] += 1
#     return res

# def cbca(np.ndarray[np.int_t, ndim=3] x0c,
#          np.ndarray[np.int_t, ndim=3] x1c,
#          np.ndarray[np.float64_t, ndim=3] vol,
#          int t):
#     cdef np.ndarray[np.float64_t, ndim=3] res
#     cdef int i, j, ii, jj, ii_s, ii_t, jj_s, jj_t, d, cnt
#     cdef double sum

#     res = np.empty_like(vol)
#     for d in range(disp_max):
#         for i in range(height):
#             for j in range(width):
#                 if j - d < 0:
#                     res[d,i,j] = vol[d,i,j]
#                     continue
#                 sum = 0
#                 cnt = 0
#                 if t:
#                     # horizontal then vertical
#                     jj_s = max(x0c[i,j,2], x1c[i,j-d,2] + d) + 1
#                     jj_t = min(x0c[i,j,3], x1c[i,j-d,3] + d)
#                     for jj in range(jj_s, jj_t):
#                         ii_s = max(x0c[i,jj,0], x1c[i,jj-d,0]) + 1
#                         ii_t = min(x0c[i,jj,1], x1c[i,jj-d,1])
#                         for ii in range(ii_s, ii_t):
#                             sum += vol[d, ii, jj]
#                             cnt += 1
#                 else:
#                     # vertical then horizontal
#                     ii_s = max(x0c[i,j,0], x1c[i,j-d,0]) + 1
#                     ii_t = min(x0c[i,j,1], x1c[i,j-d,1])
#                     for ii in range(ii_s, ii_t):
#                         jj_s = max(x0c[ii,j,2], x1c[ii,j-d,2] + d) + 1
#                         jj_t = min(x0c[ii,j,3], x1c[ii,j-d,3] + d)
#                         for jj in range(jj_s, jj_t):
#                             sum += vol[d, ii, jj]
#                             cnt += 1
#                 assert(cnt > 0)
#                 res[d, i, j] = sum / cnt
#     return res


def sgm(np.ndarray[np.float64_t, ndim=2] x0,
        np.ndarray[np.float64_t, ndim=2] x1,
        np.ndarray[np.float64_t, ndim=3] vol):
    cdef np.ndarray[np.float64_t, ndim=3] res, v0, v1, v2, v3

    cdef int i, j, d
    cdef double min_curr, min_prev, P1, P2, D1, D2
    
    # left-right
    res = np.empty_like(vol)
    min_prev = 0
    for i in range(height):
        for j in range(width):
            min_curr = INFINITY
            for d in range(disp_max):
                if j - d / disp_scale - 1 < 0:
                    res[i,j,d] = vol[i,j,d]
                else:
                    D1 = abs(x0[i,j] - x0[i,j-1])
                    D2 = abs(x1[i,j-d/disp_scale] - x1[i,j-d/disp_scale-1])
                    if   D1 <  tau_so and D2 <  tau_so: P1, P2 = pi1,      pi2
                    elif D1 <  tau_so and D2 >= tau_so: P1, P2 = pi1 / 4,  pi2 / 4
                    elif D1 >= tau_so and D2 <  tau_so: P1, P2 = pi1 / 4,  pi2 / 4
                    else:                               P1, P2 = pi1 / 10, pi2 / 10

                    res[i,j,d] = vol[i,j,d] + min(
                        res[i,j-1,d],
                        res[i,j-1,d-1] + P1 if d-1 >= 0 else INFINITY,
                        res[i,j-1,d+1] + P1 if d+1 < disp_max else INFINITY,
                        min_prev + P2) - min_prev
                if res[i,j,d] < min_curr:
                    min_curr = res[i,j,d]
            min_prev = min_curr
    v0 = res

    # right-left
    res = np.empty_like(vol)
    for i in range(height):
        for j in range(width - 1, -1, -1):
            min_curr = INFINITY
            for d in range(disp_max):
                if j + 1 >= width or j - d / disp_scale < 0:
                    res[i,j,d] = vol[i,j,d]
                else:
                    D1 = abs(x0[i,j] - x0[i,j+1])
                    D2 = abs(x1[i,j-d/disp_scale] - x1[i,j-d/disp_scale+1])
                    if   D1 <  tau_so and D2 <  tau_so: P1, P2 = pi1, pi2
                    elif D1 <  tau_so and D2 >= tau_so: P1, P2 = pi1 / 4., pi2 / 4.
                    elif D1 >= tau_so and D2 <  tau_so: P1, P2 = pi1 / 4., pi2 / 4.
                    else:                               P1, P2 = pi1 / 10, pi2 / 10

                    res[i,j,d] = vol[i,j,d] - min_prev + min(
                        res[i,j+1,d],
                        res[i,j+1,d-1] + P1 if d-1 >= 0 else INFINITY,
                        res[i,j+1,d+1] + P1 if d+1 < disp_max else INFINITY,
                        min_prev + P2)
                if res[i,j,d] < min_curr:
                    min_curr = res[i,j,d]
            min_prev = min_curr
    v1 = res

    # up-down
    res = np.empty_like(vol)
    for j in range(width):
        for i in range(height):
            min_curr = INFINITY
            for d in range(disp_max):
                if j - d / disp_scale < 0 or i - 1 < 0:
                    res[i,j,d] = vol[i,j,d]
                else:
                    D1 = abs(x0[i,j] - x0[i-1,j])
                    D2 = abs(x1[i,j-d/disp_scale] - x1[i-1,j-d/disp_scale])
                    if   D1 <  tau_so and D2 <  tau_so: P1, P2 = pi1, pi2
                    elif D1 <  tau_so and D2 >= tau_so: P1, P2 = pi1 / 4, pi2 / 4
                    elif D1 >= tau_so and D2 <  tau_so: P1, P2 = pi1 / 4, pi2 / 4
                    else:                               P1, P2 = pi1 / 10, pi2 / 10

                    res[i,j,d] = vol[i,j,d] - min_prev + min(
                        res[i-1,j,d],
                        res[i-1,j,d-1] + P1 if d-1 >= 0 else INFINITY,
                        res[i-1,j,d+1] + P1 if d+1 < disp_max else INFINITY,
                        min_prev + P2)
                if res[i,j,d] < min_curr:
                    min_curr = res[i,j,d]
            min_prev = min_curr
    v2 = res

    # down-up
    res = np.empty_like(vol)
    for j in range(width):
        for i in range(height - 1, -1, -1):
            min_curr = INFINITY
            for d in range(disp_max):
                if j - d / disp_scale < 0 or i + 1 >= height:
                    res[i,j,d] = vol[i,j,d]
                else:
                    D1 = abs(x0[i,j] - x0[i+1,j])
                    D2 = abs(x1[i,j-d/disp_scale] - x1[i+1,j-d/disp_scale])
                    if   D1 <  tau_so and D2 <  tau_so: P1, P2 = pi1, pi2
                    elif D1 <  tau_so and D2 >= tau_so: P1, P2 = pi1 / 4, pi2 / 4
                    elif D1 >= tau_so and D2 <  tau_so: P1, P2 = pi1 / 4, pi2 / 4
                    else:                               P1, P2 = pi1 / 10, pi2 / 10

                    res[i,j,d] = vol[i,j,d] - min_prev + min(
                        res[i+1,j,d],
                        res[i+1,j,d-1] + P1 if d-1 >= 0 else INFINITY,
                        res[i+1,j,d+1] + P1 if d+1 < disp_max else INFINITY,
                        min_prev + P2)
                if res[i,j,d] < min_curr:
                    min_curr = res[i,j,d]
            min_prev = min_curr
    v3 = res

    return (v0 + v1 + v2 + v3) / 4

# def outlier_detection(np.ndarray[np.int_t, ndim=2] d0, 
#                       np.ndarray[np.int_t, ndim=2] d1):
#     cdef np.ndarray[np.int_t, ndim=2] outlier
#     cdef int i, j, d

#     outlier = np.empty_like(d0)
#     for i in range(height):
#         for j in range(width):
#             if j - d0[i,j] < 0:
#                 outlier[i,j] = 2
#             elif abs(d0[i,j] - d1[i,j - d0[i,j]]) < 1.1:
#                 # not an outlier
#                 outlier[i,j] = 0
#             else:
#                 for d in range(disp_max):
#                     if j - d > 0 and abs(d - d1[i,j - d]) < 1.1:
#                         # mismatch
#                         outlier[i,j] = 1
#                         break
#                 else:
#                     # occlusion
#                     outlier[i,j] = 2
#     return outlier

# def iterative_region_voting(np.ndarray[np.int_t, ndim=3] x0c,
#                             np.ndarray[np.int_t, ndim=2] d0,
#                             np.ndarray[np.int_t, ndim=2] outlier):

#     cdef np.ndarray[np.int_t, ndim=1] hist
#     cdef np.ndarray[np.int_t, ndim=2] d0_res, outlier_res
#     cdef int i, j, k, ii, jj, d, cnt

#     hist = np.empty(disp_max, dtype=int)
#     d0_res = np.empty_like(d0)
#     outlier_res = np.empty_like(outlier)
#     for i in range(height):
#         for j in range(width):
#             d = d0[i,j]
#             d0_res[i,j] = d
#             outlier_res[i,j] = outlier[i,j]
#             if outlier[i,j] == 0:
#                 continue
#             for k in range(disp_max):
#                 hist[k] = 0
#             cnt = 0
#             for ii in range(x0c[i,j,0] + 1, x0c[i,j,1]):
#                 for jj in range(x0c[ii,j,2] + 1, x0c[ii,j,3]):
#                     if outlier[ii,jj] == 0:
#                         hist[d0[ii,jj]] += 1
#                         cnt += 1
#             d = hist.argmax()
#             if cnt > tau_s and float(hist[d]) / cnt > tau_h:
#                 outlier_res[i,j] = 0
#                 d0_res[i,j] = d
#     return d0_res, outlier_res

# def proper_interpolation(np.ndarray[np.float64_t, ndim=3] x0,
#                          np.ndarray[np.int_t, ndim=2] d0,
#                          np.ndarray[np.int_t, ndim=2] outlier):

#     cdef np.ndarray[np.float64_t, ndim=2] dir
#     cdef np.ndarray[np.int_t, ndim=2] d0_res
#     cdef int i, j, ii, jj, min_d, d
#     cdef double min_val, di, dj, ii_d, jj_d, dist

#     dir = np.array([
#         [0   ,  1],
#         [-0.5,  1],
#         [-1  ,  1],
#         [-1  ,  0.5],
#         [-1  ,  0],
#         [-1  , -0.5],
#         [-1  , -1],
#         [-0.5, -1],
#         [0   , -1],
#         [0.5 , -1],
#         [1   , -1],
#         [1   , -0.5],
#         [1   ,  0],
#         [1   ,  0.5],
#         [1   ,  1],
#         [0.5 ,  1]
#     ])

#     d0_res = np.empty_like(d0)
#     for i in range(height):
#         for j in range(width):
#             d0_res[i,j] = d0[i,j]
#             if outlier[i,j] != 0:
#                 min_val = INFINITY
#                 min_d = -1
#                 for d in range(16):
#                     dj, di = dir[d,0], dir[d,1]
#                     ii_d, jj_d = i, j
#                     ii, jj = round(ii_d), round(jj_d)
#                     while 0 <= ii < height and 0 <= jj < width and outlier[ii,jj] != 0:
#                         ii_d += di
#                         jj_d += dj
#                         ii, jj = round(ii_d), round(jj_d)
#                     if 0 <= ii < height and 0 <= jj < width:
#                         assert(outlier[ii,jj] == 0)
#                         if outlier[i,j] == 1:
#                             # mismatch
#                             dist = max(abs(x0[i,j,0] - x0[ii,jj,0]),
#                                        abs(x0[i,j,1] - x0[ii,jj,1]),
#                                        abs(x0[i,j,2] - x0[ii,jj,2]))
#                         else:
#                             # occlusion
#                             dist = d0[ii,jj]

#                         if dist < min_val:
#                             min_val = dist
#                             min_d = d0[ii,jj]
#                 assert(min_d != -1)
#                 d0_res[i,j] = min_d
#     return d0_res

def depth_discontinuity_adjustment(np.ndarray[np.int_t, ndim=2] d0,
                                   np.ndarray[np.float64_t, ndim=3] vol):
    cdef np.ndarray[np.int_t, ndim=2] d0_res, d0s
    cdef int i, j, d

    # horizontal
    d0_res = np.empty_like(d0)
    d0s = sobel(d0, 0)
    for i in range(height):
        for j in range(width):
            d0_res[i,j] = d0[i,j]
            if d0s[i,j] > tau_e and 1 <= j < width - 1:
                d = d0[i,j]
                if vol[i,j,d0[i,j-1]] < vol[i,j,d]:
                    d = d0[i,j-1]
                if vol[i,j,d0[i,j+1]] < vol[i,j,d]:
                    d = d0[i,j+1]
                d0_res[i,j]= d

    # vertical
    d0 = d0_res
    d0_res = np.empty_like(d0)
    d0s = sobel(d0, 1)
    for i in range(height):
        for j in range(width):
            d0_res[i,j] = d0[i,j]
            if d0s[i,j] > tau_e and 1 <= i < height - 1:
                d = d0[i,j]
                if vol[i,j,d0[i-1,j]] < vol[i,j,d]:
                    d = d0[i-1,j]
                if vol[i,j,d0[i+1,j]] < vol[i,j,d]:
                    d = d0[i+1,j]
                d0_res[i,j]= d

    return d0_res

def subpixel_enhancement(np.ndarray[np.int_t, ndim=2] d0,
                         np.ndarray[np.float64_t, ndim=3] vol):
    cdef np.ndarray[np.float64_t, ndim=2] d0_res
    cdef int i, j, d
    cdef double cn, cz, cp, denom

    d0_res = np.empty((height, width))
    for i in range(height):
        for j in range(width):
            d = d0[i,j]
            d0_res[i,j] = d
            if 1 <= d < disp_max - 1:
                cn = vol[i,j,d-1]
                cz = vol[i,j,d]
                cp = vol[i,j,d+1]
                denom = 2 * (cp + cn - 2 * cz)
                if denom > 1e-5:
                    d0_res[i,j] = d - min(1, max(-1, (cp - cn) / denom))
    return d0_res
