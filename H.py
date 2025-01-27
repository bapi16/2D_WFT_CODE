import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1, i0e as be,i1e,ive
import sys

def E(i,j,t,Qx,a,b):
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral, 
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'=Ax-Bx
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t  
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j==0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)
        
def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    return S1*S2*np.pi/(a+b)

def O(a,l1,A,b,l2,B):
    """this function evaluates the 1D overlap that we will use during the kinetic integrals
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
       l1: angular momentum for Gaussian 'a'
        l2: angular momentum for Gaussian 'b'
        A:    origin of Gaussian 'a'
        B:    origin of Gaussian 'b'
    """
    S1 = E(l1,l2,0,A-B,a,b)
    return S1*np.power(np.pi/(a+b),0.5)

def S(a,b):
    s = 0.0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            s+= a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*overlap(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin)
            #print(s)
    return s
     


class BasisFunction(object):
    ''' A class that contains all our basis function data
        Attributes:
        origin: array/list containing the coordinates of the Gaussian origin
        shell:  tuple of angular momentum
        exps:   list of primitive Gaussian exponents
        coefs:  list of primitive Gaussian coefficients
        norm:   list of normalization factors for Gaussian primitives
    '''
    
    def __init__(self,origin=[],shell=[],num_coefs=0,exps=[],coefs=[]):
        self.origin = np.asarray(origin)
        self.shell = np.array(shell)
        self.exps  = np.array(exps)
        self.coefs = np.array(coefs)
        self.num_coefs = num_coefs
        self.norm = np.zeros(len(self.coefs))
        self.normalize()

    def normalize(self):
        """Routine to normalize the BasisFunction objects.
           Returns self.norm, which is a list of doubles that
           normalizes the contracted Gaussian basis functions (CGBFs)

           First normalized the primitives, then takes the results and
           normalizes the contracted functions. Both steps are required,
           though I could make it one step if need be.
        """
        for i in range(self.num_coefs):
            self.norm[i]=np.power(overlap(self.exps[i],self.shell,self.origin,self.exps[i],self.shell,self.origin),-0.5)
          
            print(self.norm[i])  
          #let's normalize our contractd gaussina in our way
        N=0.0
        for i in range(self.num_coefs):
            for j in range(self.num_coefs):
                N=N+self.norm[i]*self.norm[j]*self.coefs[i]*self.coefs[j]*overlap(self.exps[i],self.shell,self.origin,self.exps[j],self.shell,self.origin)
                #print(N)
        for i in range(self.num_coefs):
            self.coefs[i] = self.coefs[i]/np.sqrt(N)
            #print(self.coefs[i])


def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a:   array of orbital exponent on Gaussian 'a' (e.g. alpha in the text) #in our case is an array since we work with anisotropic
        b:    array orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int array containing orbital angular momentum (e.g. [1,0,0]) for Gaussian 'a'
        lmn2: int array containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    nx,ny=lmn1
    mx,my=lmn2
    Ax,Ay=A
    Bx,By=B

    squarebracket1=0.5*nx*mx*O(a,nx-1,Ax,b,mx-1,Bx)-a*mx*O(a,nx+1,Ax,b,mx-1,Bx)-b*nx*O(a,nx-1,Ax,b,mx+1,Bx)+2*a*b*O(a,nx+1,Ax,b,mx+1,Bx)
    squarebracket2=0.5*ny*my*O(a,ny-1,Ay,b,my-1,By)-a*my*O(a,ny+1,Ay,b,my-1,By)-b*ny*O(a,ny-1,Ay,b,my+1,By)+2*a*b*O(a,ny+1,Ay,b,my+1,By)
   
    return O(a,ny,Ay,b,my,By)*squarebracket1+O(a,nx,Ax,b,mx,Bx)*squarebracket2
    #print(kinetic(a,lmn1,A,b,lmn2,B))
def T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    t = 0.0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            t += a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*kinetic(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin)
    return t
def R(t, u, n, p, PCx, PCy, PC,z):
    val = 0.0
    #print(u," ",t)
    if u == t == 0:
        if n == -1:
            val+=i1e(z)
        else:
            val+=ive(n,z)
    elif u == 0:
        if t > 1:
            val += -p * 0.5 * ((t-1)* (R(t - 2, u, n - 1, p, PCx, PCy, PC,z) + 2 * R(t - 2, u, n, p, PCx, PCy, PC,z) + R(t - 2, u, n + 1, p, PCx, PCy, PC,z)))  
            val += -p*0.5*PCx * (R(t-1, u, n - 1, p, PCx, PCy, PC,z) + 2 * R(t-1, u, n, p, PCx, PCy, PC,z) + R(t-1, u, n + 1, p, PCx, PCy, PC,z))
    else:
        if u > 1:
            val += -p * 0.5 * ((u-1) * (R(t, u - 2, n - 1, p, PCx, PCy, PC,z) + 2 * R(t, u - 2, n, p, PCx, PCy, PC,z) + R(t, u-2, n + 1, p, PCx, PCy, PC,z)))  
            val +=  -p*0.5*PCy * (R(t, u-1, n - 1, p, PCx, PCy, PC,z) + 2 * R(t, u-1, n, p, PCx, PCy, PC,z) + R(t, u-1, n + 1, p, PCx, PCy, PC,z))
    return val


#def gaussian_product_center(a, A, b, B):
    #return (a * A + b * B) / (a + b)

def nuclear_attraction(a, lmn1, A, b, lmn2, B, C):
    l1, m1 = lmn1
    l2, m2 = lmn2
    p = a + b
    Ax,Ay=A
    Bx,By=B
    Cx,Cy=C
    #P = gaussian_product_center(a, A, b, B) # Gaussian composite center
    Px=(a*Ax+b*Bx)/(a+b)
    Py=(a*Ay+b*By)/(a+b)
    PC=np.sqrt((Px-Cx)**2+(Py-Cy)**2)
    val = 0.0
    for t in range(l1 + l2+1):
        for u in range(m1 + m2+1):
            val += E(l1, l2, t, Ax - Bx, a, b) * E(m1, m2, u, Ay - By, a, b) * R(t, u, 0, p, Px - Cx, Py - Cy, PC, -p * PC * PC * 0.5)
    val *= np.pi * np.sqrt(np.pi / p)
    return val

def V(a,b,C):
    """this function evaluats the electron-nuclei interaction integral between two contracted gaussian
    a: is the first gaussian
    b: is the second gaussian
    C: is the origin of the nuclei"""
    N=0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            N=N+a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*nuclear_attraction(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin,C)
        #print(N)
    return N
    
def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    Ax,Ay=A
    Bx,By=B
    Cx,Cy=C
    Dx,Dy = D
   
    l1, m1 = lmn1
    l2, m2 = lmn2
    l3, m3 = lmn3
    l4, m4 = lmn4
   
    Px = (a*Ax+b*Bx)/(a+b)
    Qx = (c*Cx+d*Dx)/(c+d)
    Py = (a*Ay+b*By)/(a+b)
    Qy = (c*Cy+d*Dy)/(c+d)

    p = a + b
    q = c + d
   
    Delx = Qx - Px
    Dely = Qy - Py
   
    Delta = np.sqrt(Delx*Delx + Dely*Dely)
    sigma = (p+q)/(4*p*q)
   
    #val = 0
    temp1 = 0
    #temp2 = 0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for tp in range(l3+l4+1):
                for up in range(m3+m4+1):
                    temp1+= ((np.power(np.pi, 2))/(p*q))*np.sqrt(np.pi/(4*sigma))*E(l1,l2,t,A[0]-B[0],a,b) * E(m1,m2,u,A[1]-B[1],a,b) * np.power(-1,t+u)*E(l3,l4,tp,C[0]-D[0],c,d) * E(m3,m4,up,C[1]-D[1],c,d) * R(t+tp,u+up,0,(1/(4*sigma)),Delx,Dely,Delta,(-1)*(Delta**2)/(8*sigma))

    
    return temp1

def U(a,b,c,d):
    N=0
    for i in range(len(a.coefs)):
        for j in range(len(b.coefs)):
            for k in range(len(c.coefs)):
                for l in range(len(d.coefs)):
                    N += a.norm[i]*b.norm[j]*a.coefs[i]*b.coefs[j]*c.norm[k]*d.norm[l]*c.coefs[k]*d.coefs[l]*electron_repulsion(a.exps[i],a.shell,a.origin,b.exps[j],b.shell,b.origin,c.exps[k],c.shell,c.origin,d.exps[l],d.shell,d.origin)
        #print(N)
    return N

###################################Recipie for Basis set generation##################################################################
def create_basis(origin, shell_type, n_basis, start_exp, factor):
    """
    Automate the generation of basis functions, supporting s, p, d, and f shells.
    
    Args:
        origin (list): Origin of the basis function (e.g., [0.0, -0.05]).
        shell_type (str): Type of shell ('s', 'p', 'd', 'f').
        n_basis (int): Number of basis functions to generate.
        start_exp (float): Starting exponent value for the basis set.
        factor (float): Multiplicative factor for the exponents.
    
    Returns:
        list: A list of BasisFunction objects.
    """
    basis_set = []
    shell_map = {
        's': [[0, 0]],
        'p': [[1, 0], [0, 1]],
        'd': [[2, 0], [1, 1], [0, 2]],  # Example combinations for d-shell
        'f': [[3, 0], [2, 1], [1, 2], [0, 3]]  # Example combinations for f-shell
    }

    if shell_type not in shell_map:
        raise ValueError("Invalid shell type. Use 's', 'p', 'd', or 'f'.")

    shells = shell_map[shell_type]
    for i in range(n_basis):
        my_origin = np.array(origin)
        my_coefs = np.array([1])
        my_exps = np.array([start_exp * np.power(factor, i)])
        num_coefs = len(my_coefs)
        shell = np.array(shells[i % len(shells)])  # Cycle through shell types if needed
        basis_set.append(
            BasisFunction(origin=my_origin, shell=shell, num_coefs=num_coefs, exps=my_exps, coefs=my_coefs)
        )
    return basis_set


# Generate the basis sets
BASIS = []

# Basis set for atom at (0, 0.0) with s-type shells
BASIS.extend(create_basis(origin=[0, 0.0], shell_type='s', n_basis=32, start_exp=0.006, factor=2))

# Basis set for atom at (0, 0.0) with p-type shells
#BASIS.extend(create_basis(origin=[0, 0.0], shell_type='p', n_basis=4, start_exp=0.05, factor=2))

######################################xxxxx#############################################################################################


n = len(BASIS)
atom_origin=[[0.0000,  0.0],
                [0.00, 0.0]]
S_matrix = np.zeros([n, n])
T_matrix=np.zeros([n,n])
V_matrix=np.zeros([n,n])
G_matrix = np.zeros((n, n, n, n))


Z1=2
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                S_matrix[i, j] = S(BASIS[i], BASIS[j])
                T_matrix[i, j] = T(BASIS[i], BASIS[j])
                V_matrix[i, j] = -(Z1*V(BASIS[i], BASIS[j],np.array(atom_origin[0])))
                G_matrix[i, j, k, l] = G_matrix[l, k, j, i] = U(BASIS[i], BASIS[j], BASIS[k], BASIS[l])
                #U_matrix[i,j,k,l] = U(BASIS[i], BASIS1[j], BASIS[k], BASIS1[l])


import pyscf
import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1
from scipy.special import comb
from scipy.integrate import quad
from pyscf.gto.basis import parse_gaussian 
from numpy import load, einsum, dot
from pyscf import gto, scf, dft, cc, mp, ao2mo, ci, fci


#######Defining the system and methods###############

mol = gto.M(atom='''He 0.0 0.0 '''
            , spin=0,  cart=True, symmetry = False)

mol.basis = { 'He' : #parse_gaussian.load('basis_set.gbs','Li')}
"""unc    
He S

0.1 1.0
0.2 1.0
0.3 1.0
0.4 1.0

0.5 1.0
0.6 1.0
0.7 1.0
0.8 1.0

0.9 1.0
1.0 1.0
2.0 1.0
3.0 1.0

4.0 1.0
5.0 1.0
6.0 1.0
7.0 1.0

8.0 1.0
9.0 1.0
10.0 1.0
11.0 1.0

12.0 1.0
13.0 1.0
14.0 1.0
15.0 1.0

16.0 1.0
17.0 1.0
18.0 1.0
19.0 1.0

20.0 1.0
21.0 1.0
22.0 1.0
23.0 1.0


"""
}

mol.build()
#mol.verbose = 4



mol.incore_anyway = True

print(' ****** 2D RESULTS ****')
rhf_mf  = rhf_mf = scf.UHF(mol).apply(scf.addons.remove_linear_dep_)
#mf.nelec = (1, 0)

####overwrite pyscf values with my code 

rhf_mf.max_cycle=2000
rhf_mf.get_ovlp = lambda *args: S_matrix
rhf_mf._eri = ao2mo.restore(1, G_matrix, S_matrix.shape[0])
#rhf_mf._eri = ao2mo.restore(1, G_matrix, S_matrix.shape[0])
H1 = T_matrix +  V_matrix 
rhf_mf.get_hcore = lambda *args: H1
rhf_mf = scf.addons.remove_linear_dep_(rhf_mf, threshold=1e-5, lindep=1e-5)
rhf_mf.run()
r=rhf_mf.run()

# Retrieve the total energy after the RHF calculation has completed


mymp = mp.MP2(rhf_mf).run()

mycc = cc.CCSD(rhf_mf).run()
et = mycc.ccsd_t() # works only
#mycc = cc.CCSD(rhf_mf)
#mycc.run()
#ccsd_energy = mycc.kernel()
#print(ccsd_energy)
# Perform (T) correction
#ccsd_t_energy = mycc.ccsd_t()
#ccsdt=ccsd_energy+ccsd_t_energy  
#print("b",ccsdt)

cisolver = fci.FCI(rhf_mf)
e_fci, _ = cisolver.kernel()  # Run FCI calculation and get the energy
print("fci=",e_fci)

