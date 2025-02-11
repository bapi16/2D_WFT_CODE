import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1, i0e as be, i1e, ive
from multiprocessing import Pool, cpu_count, Array
import sys
import itertools, time
from scipy.linalg import eigh
from multiprocessing import RawArray
# ---------------------------
# Core Integral Implementations
# ---------------------------
#start time
start_time = time.time()

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

'''# ---------------------------
# Basis Function Class
# ---------------------------
class BasisFunction:
    def __init__(self, origin, shell, exps, coefs):
        self.origin = np.array(origin)
        self.shell = np.array(shell)
        self.exps = np.array(exps)
        self.coefs = np.array(coefs)
        self.norm = np.zeros_like(coefs)
        self.normalize()

    def normalize(self):
        # Normalize each primitive
        for i in range(len(self.exps)):
            S = overlap(self.exps[i], self.shell, self.origin, 
                        self.exps[i], self.shell, self.origin)
            self.norm[i] = 1/np.sqrt(S)
        
        # Normalize contraction
        N = 0.0
        for i in range(len(self.exps)):
            for j in range(len(self.exps)):
                S = overlap(self.exps[i], self.shell, self.origin,
                            self.exps[j], self.shell, self.origin)
                N += self.norm[i]*self.norm[j]*self.coefs[i]*self.coefs[j]*S
        self.coefs /= np.sqrt(N)'''

'''# ---------------------------
# Basis Function Class
# ---------------------------
class BasisFunction:
    def __init__(self, origin, shell, exps, coefs):
        self.origin = np.array(origin)
        self.shell = np.array(shell)
        self.exps = np.array(exps)
        self.coefs = np.array(coefs)
        self.norm = np.zeros_like(coefs)
        self.normalize()

    def normalize(self):
        for i in range(len(self.exps)):
            S = overlap(self.exps[i], self.shell, self.origin,
                        self.exps[i], self.shell, self.origin)
            self.norm[i] = 1 / np.sqrt(S)
        N = sum(self.norm[i] * self.norm[j] * self.coefs[i] * self.coefs[j] *
                overlap(self.exps[i], self.shell, self.origin,
                       self.exps[j], self.shell, self.origin)
                for i in range(len(self.exps)) for j in range(len(self.exps)))
        self.coefs /= np.sqrt(N)'''

# ---------------------------
# Molecular Components
# ---------------------------
class Atom:
    _atomic_data = {
        'H': {'Z': 1}, 'He': {'Z': 2}, 'Li': {'Z': 3}, 'Be': {'Z': 4},
        'B': {'Z': 5}, 'N': {'Z': 6}, 'F': {'Z': 7}, 'Ne': {'Z': 8},
        'Na': {'Z': 9}, 'Mg': {'Z': 10}
    }
    def __init__(self, symbol, position, charge=0):
        self.symbol = symbol
        self.position = np.array(position)
        self.Z = self._atomic_data[symbol]['Z'] - charge

def calculate_nuclear_repulsion(atoms):
    E_nuc = 0.0
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            R = np.linalg.norm(atoms[i].position - atoms[j].position)
            E_nuc += atoms[i].Z * atoms[j].Z / R
    return E_nuc

# ---------------------------
# Basis Set Creation (Corrected)
# ---------------------------

def create_basis(origin, shell_type, n_basis, start_exp, factor):
    shell_map = {
        's': [[0, 0]],          # 1 component
        'p': [[1, 0], [0, 1]],  # 2 components
        'd': [[2, 0], [1, 1], [0, 2]],          # 3 components
        'f': [[3, 0], [2, 1], [1, 2], [0, 3]]   # 4 components
    }
    basis_set = []
    
    # Generate exponents first
    exponents = [start_exp * (factor ** i) for i in range(n_basis)]
    
    # Create basis functions: all angular components for each exponent
    for exp in exponents:
        for angular_comp in shell_map[shell_type]:
            basis_set.append(
                BasisFunction(
                    origin=np.array(origin),
                    shell=np.array(angular_comp),
                    num_coefs=1,
                    exps=np.array([exp]),
                    coefs=np.array([1])
                )
            )
    
    return basis_set


# ---------------------------
# Parallel Integral Computation
# ---------------------------
def compute_STV_chunk(args):
    start, end, basis, n, atoms = args
    for idx in range(start, end):
        i, j = divmod(idx, n)
        S_val = T_val = V_val = 0.0
        for p in range(len(basis[i].exps)):
            for q in range(len(basis[j].exps)):
                a = basis[i].exps[p]; b = basis[j].exps[q]
                A = basis[i].origin; B = basis[j].origin
                shell_i = basis[i].shell; shell_j = basis[j].shell
                coef = basis[i].norm[p] * basis[j].norm[q] * basis[i].coefs[p] * basis[j].coefs[q]
                S_val += coef * overlap(a, shell_i, A, b, shell_j, B)
                T_val += coef * kinetic(a, shell_i, A, b, shell_j, B)
                for atom in atoms:
                    C = atom.position
                    V_val -= atom.Z * coef * nuclear_attraction(a, shell_i, A, b, shell_j, B, C)
        S_matrix[i,j] = S_val; T_matrix[i,j] = T_val; V_matrix[i,j] = V_val

def compute_G_chunk(args):
    start, end, basis, n = args
    for idx in range(start, end):
        i, j, k, l = np.unravel_index(idx, (n,n,n,n))
        integral = 0.0
        for p in range(len(basis[i].exps)):
            for q in range(len(basis[j].exps)):
                for r in range(len(basis[k].exps)):
                    for s in range(len(basis[l].exps)):
                        a = basis[i].exps[p]; b = basis[j].exps[q]
                        c = basis[k].exps[r]; d = basis[l].exps[s]
                        A = basis[i].origin; B = basis[j].origin
                        C = basis[k].origin; D = basis[l].origin
                        coef = (basis[i].norm[p] * basis[j].norm[q] *
                                basis[k].norm[r] * basis[l].norm[s] *
                                basis[i].coefs[p] * basis[j].coefs[q] *
                                basis[k].coefs[r] * basis[l].coefs[s])
                        integral += coef * electron_repulsion(
                            a, basis[i].shell, A, b, basis[j].shell, B,
                            c, basis[k].shell, C, d, basis[l].shell, D)
        G_matrix[i,j,k,l] = integral

# ---------------------------
# Main Program
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()

    # ==== Molecular Configuration ====
    atoms = [Atom('H', [0.0, 0.0]),Atom('Li', [0.0, 0.1])]  # H2 molecule

    # ==== Basis Set Creation ====
    BASIS = []
    for atom in atoms:
        if atom.symbol == 'H':
            BASIS.extend(create_basis(atom.position, 's', 32, 0.006, 2))
        elif atom.symbol=='Li':
            BASIS.extend(create_basis(atom.position, 's', 36, 0.0005, 2))
    n = len(BASIS)
    print(f"Basis functions: {n}")

    # ==== Shared Memory Setup ====


    # Initialize with RawArray instead of Array
    S_flat = RawArray('d', n*n)
    T_flat = RawArray('d', n*n)
    V_flat = RawArray('d', n*n)
    G_flat = RawArray('d', n**4)

    # Direct buffer access without get_obj()
    S_matrix = np.frombuffer(S_flat, dtype=np.float64).reshape(n,n)
    T_matrix = np.frombuffer(T_flat, dtype=np.float64).reshape(n,n)
    V_matrix = np.frombuffer(V_flat, dtype=np.float64).reshape(n,n)
    G_matrix = np.frombuffer(G_flat, dtype=np.float64).reshape(n,n,n,n)

    # ==== Compute Integrals ====
    print("Computing integrals...")
    with Pool(cpu_count()) as pool:
        # S, T, V
        total = n*n
        chunks = [(i, min(i+1000, total), BASIS, n, atoms) for i in range(0, total, 1000)]
        pool.map(compute_STV_chunk, chunks)
        # G
        total = n**4
        chunks = [(i, min(i+1000, total), BASIS, n) for i in range(0, total, 1000)]
        pool.map(compute_G_chunk, chunks)

    # ==== SCF Setup ====
    H_core = T_matrix + V_matrix
    S = S_matrix
    E_nuc = calculate_nuclear_repulsion(atoms)
    n_electrons = sum(atom.Z for atom in atoms)
    # Determine if unrestricted based on electron count
    unrestricted = (n_electrons % 2 != 0)  # Auto-set UHF for odd electrons

    if unrestricted:
        n_alpha = (n_electrons + 1) // 2
        n_beta = n_electrons // 2
        print(f"\nUHF Calculation: {n_alpha} alpha, {n_beta} beta electrons")

        # Initial guess using core Hamiltonian
        _, C_alpha = eigh(H_core, S)
        C_beta = C_alpha.copy()
        D_alpha = C_alpha[:, :n_alpha] @ C_alpha[:, :n_alpha].T
        D_beta = C_beta[:, :n_beta] @ C_beta[:, :n_beta].T

        print("Iter | Energy (Ha) | ΔE (Ha) | ΔD")
        E_old = 0.0
        for iter in range(50):
            D_total = D_alpha + D_beta
            J = np.einsum('ijkl,kl->ij', G_matrix, D_total)
            K_alpha = np.einsum('ikjl,kl->ij', G_matrix, D_alpha)
            K_beta = np.einsum('ikjl,kl->ij', G_matrix, D_beta)

            F_alpha = H_core + J - K_alpha
            F_beta = H_core + J - K_beta

            # Diagonalize
            _, C_alpha = eigh(F_alpha, S)
            _, C_beta = eigh(F_beta, S)

            # New densities
            D_alpha_new = C_alpha[:, :n_alpha] @ C_alpha[:, :n_alpha].T
            D_beta_new = C_beta[:, :n_beta] @ C_beta[:, :n_beta].T

            # Energy calculation
            energy_one = np.sum((D_alpha + D_beta) * H_core)
            energy_J = 0.5 * np.sum(D_total * J)
            energy_K = -0.5 * (np.sum(D_alpha * K_alpha) + np.sum(D_beta * K_beta))
            energy = energy_one + energy_J + energy_K + E_nuc
            ΔE = abs(energy - E_old)
            ΔD = max(np.max(np.abs(D_alpha_new - D_alpha)), 
                   np.max(np.abs(D_beta_new - D_beta)))

            print(f"{iter:4d} | {energy:.6f} | {ΔE:.2e} | {ΔD:.2e}")
            if ΔE < 1e-6 and ΔD < 1e-6:
                break

            D_alpha, D_beta = D_alpha_new, D_beta_new
            E_old = energy

    else:  # RHF
        n_occ = n_electrons // 2
        print(f"\nRHF Calculation: {n_occ} doubly occupied orbitals")

        # Initial guess using core Hamiltonian
        _, C = eigh(H_core, S)
        D = 2 * C[:, :n_occ] @ C[:, :n_occ].T

        print("Iter | Energy (Ha) | ΔE (Ha)")
        E_old = 0.0
        for iter in range(50):
            J = np.einsum('ijkl,kl->ij', G_matrix, D)
            K = np.einsum('ikjl,kl->ij', G_matrix, D)
            F = H_core + J - 0.5 * K

            energy = np.sum(D * (H_core + 0.5*J - 0.25*K)) + E_nuc
            ΔE = abs(energy - E_old)

            print(f"{iter:4d} | {energy:.6f} | {ΔE:.2e}")
            if ΔE < 1e-6:
                break

            # Update density
            _, C = eigh(F, S)
            D = 2 * C[:, :n_occ] @ C[:, :n_occ].T
            E_old = energy

        # ==== Final Energy Output ====
    if unrestricted:
        electronic_energy = energy_one + energy_J + energy_K
        total_energy = electronic_energy + E_nuc
        print(f"\nElectronic Energy (Hα): {electronic_energy:.8f} Ha")
        print(f"Nuclear Repulsion Energy: {E_nuc:.8f} Ha")
        print(f"Total Energy (UHF): {total_energy:.8f} Ha")
    else:
        electronic_energy = np.sum(D * (H_core + 0.5*J - 0.25*K))
        total_energy = electronic_energy + E_nuc
        print(f"\nElectronic Energy (RHF): {electronic_energy:.8f} Ha")
        print(f"Nuclear Repulsion Energy: {E_nuc:.8f} Ha")
        print(f"Total Energy (RHF): {total_energy:.8f} Ha")
        
#end time
end_time = time.time()
total_time=end_time-start_time
# Print the total computation time
print(f"Total computation time: {total_time:.2f} seconds")

