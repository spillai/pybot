import numpy as np
from sympy import symbols, Matrix 
import matplotlib.pyplot as pyplot

# Continuous definition of functions (using indicator functions)
def indicator_above(x,vmin):
    x1 = Abs(vmin - x)
    return (x + x1 - vmin) / (2*x1)

def indicator_below(x,vmax):
    x1 = Abs(vmax - x)
    return 1 - (x + x1 - vmax) / (2*x1)

def indicator_range(x,vmin,vmax):
    x0 = vmax - x
    x1 = Abs(vmin - x)
    x2 = Abs(x0)
    return (x0 + x2) * (x + x1 - vmin) / (4*x1*x2)

def BS(i,k,t):
    if (k==1):
        return indicator_range(t,i,i+1)
    else:
        return BS(i,k-1,t) * (t - i)/((i+k-1) - i) + BS(i+1,k-1,t) * (i+k - t) / (i+k - (i+1));

def CBS(i,k,t):
    ind = indicator_below(t,i+k)
    s = sum( [BS(j,k,t) for j in range(i,i+k)] )
    return ind * s + (1-ind)

#t = symbols('t')
#plot(B(0,4,t), B(1,4,t), B(2,4,t), B(3,4,t), (t,0,7))
#plot(CB(0,4,t), CB(1,4,t), CB(2,4,t), CB(3,4,t), (t,0,7))

# Straightforward definition

def BS(i,k,t):
    if (k==1):
        if( i <= t and t < i+1):
            return 1.0
        else:
            return 0.0
    else:
        return BS(i,k-1,t) * (t - i)/((i+k-1) - i) + BS(i+1,k-1,t) * (i+k - t) / (i+k - (i+1))

def CBS(i,k,t):
    if(t > i+k):
        return 1.0
    else:
        return sum( [BS(j,k,t) for j in range(i,i+k)] )

# xs = np.linspace(0,7,100)
# for i in range(0,4):
#     pyplot.plot(xs, [BS(i,4,x) for x in xs])
    
# pyplot.figure(2)
# for i in range(0,4):
#     pyplot.plot(xs, [CBS(i,4,x) for x in xs])
# pyplot.show(block=True)

# from sympy import *
# u = symbols('u')
# #cse for common subexpr

# # Matrix from General matrix representations for B-splines, Quin
# M = Matrix([
#         [1, -3,  3, -1],
#     [4,  0, -6,  3],
#     [1,  3,  3, -3],
#     [0,  0,  0,  1]
#     ]) / 6.0

# Sum = Matrix((
#         [-1, 0, 0],
#     [-1,-1, 0],
#     [-1,-1,-1]
#     ))

# C = Sum * M[0:3,:]

# def U(u):
#     return Matrix(([[1],[u],[u**2], [u**3]]))

# def B(u):
#     return M * U(u)

# def B2(u):
#     return Matrix([
#         [BS(-2,4,u)],
#      [BS(-1,4,u)],
#      [BS(0,4,u)] 
#     ])

# def CB(u):
#     return Matrix(([[1],[1],[1]])) + C * U(u)

# def CB2(u):
#     return Matrix([
#         [CBS(-2,4,u)],
#      [CBS(-1,4,u)],
#      [CBS(0,4,u)] 
#     ])

# def dCB(u):
#     return CB(u).diff(u,1)

# def d2CB(u):
#     return CB(u).diff(u,2)

# plot(B(u)[0], B(u)[1], B(u)[2], (u,0,1))
# #plot(B2(u)[0], B2(u)[1], B2(u)[2], (u,0,1))
# plot(CB(u)[0], CB(u)[1], CB(u)[2], (u,0,1))
# #plot(CB2(u)[0], CB2(u)[1], CB2(u)[2], (u,0,1))
# for f in [CB, dCB, d2CB]:
#     pprint( f(u).subs(u,0.0).transpose() )
