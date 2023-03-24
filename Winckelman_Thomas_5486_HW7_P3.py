

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse as sp
from scipy.sparse.linalg import use_solver  # optional
from scipy.sparse.linalg import spsolve as solve

%matplotlib notebook


def Holt(m):        # named for the guy who showed me how to correctly use the Kronecker product
    M = m**2
    A = sp.diags( [-1, 4, -1], [-1, 0, 1], shape=(m,m) )
    A = sp.kron( sp.eye(m), A )
    A += sp.diags( [-1,-1], [-m,m], shape=(M,M) )
    return sp.csr_matrix(A)



def PrintMat(mat, roun=4, tab=None):  # still, one of my better works, and one of my first in python
    if tab is None:
        tabulate=False
    else:
        tabulate=True
    lengths = []
    m = len(mat)
    n = len(mat[0])
    for i in range(0,m):
        for j in range(0,n):
            lengths.append( len(str(round(mat[i][j], roun))) )
    space = 1+max(lengths)
    z = [[0] * n for i in range(m)]
    for i in range(m):
        for j in range(n):
            z[i][j] = round(mat[i][j], roun)
    print('')
    for i in range(0,m):
        x = str(z[i])
        x = x.lstrip('[')
        x = x.rstrip(']')
        x = x.split(', ')
        if tabulate:
            y = '   '.join([' ', x[0].rjust(tab)])
        else:
            y = '   '.join([' ', x[0].rjust(space)])
        for j in range(1, len(x)):
            y = ' '.join([y, x[j].rjust(space)])
        print(y)
    print('')


def BasicSurf(x, y, z):     # z is a len(x)-by-len(y) matrix where z[i,j] = f(x[i],x[j])
    fig = go.Figure(go.Surface(x=x, y=y, z=z))
    fig.show(renderer="colab")


#
#   solve -f = u_{xx}+u_{yy} for a function u on [a,b]X[a,b] (a Cartesian product)
#   subject to the boundary condition that (s==0 or t==0) implies u(s,t)==0
#

def temp(res, f=1, a=0, b=1, plot=False):
    res -= 1                    # I got confused about the indices and this is to correct ex post
    x = np.linspace(a, b, res+2)
    if isinstance(f, int):
        F = f*np.ones(res**2)
    else:
        F = np.zeros(( res, res ))
        x = x[1:(res+1)]
        for i in range(res):
            for j in range(res):
                F[i][j] = f(x[i],x[j])
        x = np.hstack((a,x,b))
        F = np.reshape(F, (1,-1)).T
    F = solve(
            (res+1)**2* Holt(res) , F
        ).reshape(res,res)/(b-a)
    F = np.hstack((
            np.zeros((res+2,1)),
            np.vstack(( np.zeros(res), F, np.zeros(res) )),
            np.zeros((res+2,1))
        ))
    if plot:
        BasicSurf(x,x,F)
    return x,F


N = 201
stl = N-2   # 'stl' : 'second to last'
data = []
for n in tqdm(range(4,N,2)):
    data.append([
            n,
            temp( n, plot=bool(max(0,n-stl)) )[1].max()
        ])


data = np.array(data)
t = 0.07367129792   # maximum of the true solution function
a = np.log(abs(t-data[0,1]))+np.log(data[0,0])
b = np.log(abs(t-data[0,1]))+2*np.log(data[0,0])
c = np.log(abs(t-data[0,1]))+3*np.log(data[0,0])
d = 0.06


fig,ax = plt.subplots()
fig.set_tight_layout(True)
ax.plot(
    np.log(data[:,0]),
    np.log(abs(t-data[:,1])),
    color='blue',
    label='Observed Data'
    )
ax.plot(
    np.log(data[:,0]),
    a-np.log(data[:,0])+d,
    '--',
    linewidth=1,
    markersize=1.5,
    color='red',
    label='$\mathcal{O}(N)$'
    )
ax.plot(
    np.log(data[:,0]),
    b-2*np.log(data[:,0])+d,
    '--',
    linewidth=1,
    markersize=1.5,
    color='yellow',
    label='$\mathcal{O}(N^2)$'
    )
ax.plot(
    np.log(data[:,0]),
    c-3*np.log(data[:,0])+d,
    '--',
    linewidth=1,
    markersize=1.5,
    color='green',
    label='$\mathcal{O}(N^3)$'
    )
#ax.set_xscale('log')
#ax.set_yscale('log')
plt.legend(
    facecolor='gainsboro',
    edgecolor='black',
    framealpha=1
    )
ax.set_xlabel('$\ln(N)$ (With $N$ Defined as in the Statement of the Algorithm)')
ax.set_ylabel('$\ln(|u_h(0.5,0.5)-u(0.5,0.5)|)$')
ax.set_title('log-log: Absolute Error at the Point $(0.5,0.5)$ Against $N$')
ax.grid()
plt.show()


for j in range(3,11):
    ('   N=' + str(j) + '   ').center(30, '-')
    PrintMat(
            temp(
                j, plot=bool(max(0,j-9))
                )[1], tab=4
            )



