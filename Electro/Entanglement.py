import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from qutip import basis, tensor, ket2dm, ptrace, entropy_vn

np.set_printoptions(precision=4, suppress=True)

SAVE_PLOTS = False
OUTDIR = "lab_outputs"
if SAVE_PLOTS:
    os.makedirs(OUTDIR, exist_ok=True)

zero = basis(2,0)
one  = basis(2,1)

def purity(rho):
    return rho.purity()

def zero_state(N):
    return tensor([zero]*N)

def one_state(N):
    return tensor([one]*N)

def vn_entropy_subset(rho, keep):
    return entropy_vn(ptrace(rho, keep), base=2)

def ghz_state(N):
    return (zero_state(N) + one_state(N)).unit()

def w_state(N):
    terms=[]
    for i in range(N):
        k=[zero]*N
        k[i]=one
        terms.append(tensor(k))
    return sum(terms).unit()

# Activity 5

def entropy_map_2body(rho,N):
    M=np.zeros((N,N))
    for i in range(N):
        M[i,i]=vn_entropy_subset(rho,[i])
    for i in range(N):
        for j in range(N):
            if i!=j:
                M[i,j]=vn_entropy_subset(rho,[i,j])
    return M

def plot_entropy_grid(N_list,title):
    fig,axs=plt.subplots(len(N_list),3,figsize=(9,9))

    for r,N in enumerate(N_list):
        states=[
            ("Product", ket2dm(zero_state(N))),
            ("GHZ", ket2dm(ghz_state(N))),
            ("W", ket2dm(w_state(N)))
        ]

        for c,(name,rho) in enumerate(states):
            M=entropy_map_2body(rho,N)

            # FIXED COLOR SCALE (teacher hint)
            im=axs[r,c].imshow(M, vmin=0.0, vmax=1.0)

            axs[r,c].set_title(f"{name} (N={N})")
            plt.colorbar(im,ax=axs[r,c],fraction=0.046)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Activity 6

def avg_entropy_for_k(rho,N,k):
    vals=[]
    for keep in combinations(range(N),k):
        vals.append(vn_entropy_subset(rho,list(keep)))
    return np.mean(vals)

def plot_avg_entropy_vs_k():
    N=8
    rho_g=ket2dm(ghz_state(N))
    rho_w=ket2dm(w_state(N))
    ks=range(1,N)
    g=[avg_entropy_for_k(rho_g,N,k) for k in ks]
    w=[avg_entropy_for_k(rho_w,N,k) for k in ks]

    plt.figure(figsize=(6,4))
    plt.plot(ks,g,'o-',label="GHZ")
    plt.plot(ks,w,'s-',label="W")
    plt.xlabel("Subsystem size k")
    plt.ylabel("Avg VN entropy")
    plt.title("Activity 6: Scaling (N=8)")
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

# Activity 7

def remove_one_qubit(rho,N):
    return ptrace(rho,list(range(1,N)))

def plot_reduced_density():
    for Ns in [(5,6),(7,8)]:
        fig,axs=plt.subplots(2,2,figsize=(8,8))

        for r,N in enumerate(Ns):
            rho_g=remove_one_qubit(ket2dm(ghz_state(N)),N)
            rho_w=remove_one_qubit(ket2dm(w_state(N)),N)

            # FIXED SCALE for density matrices
            im1 = axs[r,0].imshow(np.abs(rho_g.full()), vmin=0.0, vmax=0.5)
            axs[r,0].set_title(f"Reduced GHZ (N={N})")
            plt.colorbar(im1,ax=axs[r,0],fraction=0.046)

            im2 = axs[r,1].imshow(np.abs(rho_w.full()), vmin=0.0, vmax=0.5)
            axs[r,1].set_title(f"Reduced W (N={N})")
            plt.colorbar(im2,ax=axs[r,1],fraction=0.046)

        plt.tight_layout()
        plt.show()

# Main

def main():

    print("=== Activity 1 ===")
    for N in range(2,9):
        print(N, zero_state(N).shape)

    print("\n=== Activity 2 ===")
    for N in range(4,9):
        rho=ket2dm(zero_state(N))
        print(N,"Global purity",purity(rho),"Qubit purity",purity(ptrace(rho,[0])))

    print("\n=== Activity 3 ===")
    bell=(tensor(zero,zero)+tensor(one,one)).unit()
    rho=ket2dm(bell)
    print("Bell subsystem entropy:",entropy_vn(ptrace(rho,[0]),base=2))

    print("\n=== Activity 5 ===")
    plot_entropy_grid([3,4,5],"Figure 1: Two-body entropy heatmaps")
    plot_entropy_grid([6,7,8],"Figure 2: Two-body entropy heatmaps")

    print("\n=== Activity 6 ===")
    plot_avg_entropy_vs_k()

    print("\n=== Activity 7 ===")
    plot_reduced_density()

if __name__=="__main__":
    main()