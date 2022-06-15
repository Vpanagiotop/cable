import numpy as np
from scipy import integrate, optimize

L = 1000
p = 150
g = 150
E = 1.3e8
A = np.pi*0.7**2/4

def VerticalForce(x, p, L):
    return p * L/2 - p * x

def HorizontalForce(p, z, L):
    return  p * L**2 / (8 * z)

def TensionForce(x, p, z, L):
    V = VerticalForce(x, p, L)
    H = HorizontalForce(p, z, L)
    return np.sqrt(H**2 + V**2)

def Stress(x, p, z, L, A):
    return TensionForce(x, p, z, L)/A/1000

def DeformedLength(x, p, z, L):
    return integrate.quad(TensionForce, 0, x, args = (p, z, L))[0]/HorizontalForce(p, z, L)

def InitialLength(p, z, L, A):
    return DeformedLength(L, p, z, L)/(1+TensionForce(L, p, z, L)/(E*A))
def Length(z, L):
    Li = 0.5 *np.sqrt(L**2 + 16*z**2) + L**2 / (8 * z) * (
        np.log((4 * z + np.sqrt(L**2 + 16*z**2))/L))
    return Li

def Deflection(p, f, L, A):
    def TensionDeformation(p, z, A):
        return TensionForce(0, p - g, z, L)/(E*A)
    
    def LengthDeformation(z, f, L):
        return (Length(z, L) - Length(f, L))/Length(f, L)
    
    def Constrains(x, L):
        return  TensionDeformation(p, x[0], L, A) - LengthDeformation(x[0], f, L)
    
    res = optimize.root(Constrains, [1], method='broyden1', tol=1e-14)
    return abs(f-res.x[0])
    
def Plot(x, f, L):
    V = VerticalForce(x, 1, L)
    H = 1 * L**2 / (8 * f)
    return x/H * (V + 1 * x/2)