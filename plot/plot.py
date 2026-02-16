import numpy as np

import numba 

from scipy import integrate, linalg, optimize,special
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

import matplotlib.pylab as plt
import matplotlib.colors as colors

import math 

from matplotlib.path import Path
from matplotlib.patches import PathPatch

pi=np.arccos(-1)


##### Functions #####
#####################
def diag_mat(matrix,N_eigvec):
    matrix_sparse=csr_matrix(matrix)
    eigenvalue, eigenvector = eigs(matrix_sparse, k=N_eigvec, which='LM')
    return np.real(eigenvalue), np.real(eigenvector)

@numba.njit(fastmath=True)
def interp2d_numba(x1, x2, X_grid, f_vals):
    NX = X_grid.shape[0]

    # Find indices
    ix1 = np.searchsorted(X_grid, x1) - 1
    ix2 = np.searchsorted(X_grid, x2) - 1

    # Clamp to valid range
    ix1 = max(0, min(ix1, NX - 2))
    ix2 = max(0, min(ix2, NX - 2))

    # Get surrounding grid points
    x10, x11 = X_grid[ix1], X_grid[ix1 + 1]
    x20, x21 = X_grid[ix2], X_grid[ix2 + 1]

    # Compute weights
    dx1 = (x1 - x10) / (x11 - x10) if x11 > x10 else 0.0
    dx2 = (x2 - x20) / (x21 - x20) if x21 > x20 else 0.0

    # Bilinear interpolation
    f00 = f_vals[ix1, ix2]
    f10 = f_vals[ix1 + 1, ix2]
    f01 = f_vals[ix1, ix2 + 1]
    f11 = f_vals[ix1 + 1, ix2 + 1]

    interp_val = (
        (1 - dx1) * (1 - dx2) * f00 +
        dx1 * (1 - dx2) * f10 +
        (1 - dx1) * dx2 * f01 +
        dx1 * dx2 * f11
    )

    return interp_val

@numba.njit(fastmath=True)
def interp3d_numba(y, x1, x2, Y_grid, X_grid, f_vals):
    NY, NX, _ = f_vals.shape

    # Find indices in Y
    iy = np.searchsorted(Y_grid, y) - 1
    ix1 = np.searchsorted(X_grid, x1) - 1
    ix2 = np.searchsorted(X_grid, x2) - 1

    iy = max(0, min(iy, NY - 2))
    ix1 = max(0, min(ix1, NX - 2))
    ix2 = max(0, min(ix2, NX - 2))

    # Compute local coordinates
    y0, y1 = Y_grid[iy], Y_grid[iy+1]
    x10, x11 = X_grid[ix1], X_grid[ix1+1]
    x20, x21 = X_grid[ix2], X_grid[ix2+1]

    dy = (y - y0) / (y1 - y0) if y1 > y0 else 0.0
    dx1 = (x1 - x10) / (x11 - x10) if x11 > x10 else 0.0
    dx2 = (x2 - x20) / (x21 - x20) if x21 > x20 else 0.0

    # Trilinear interpolation
    def val(i, j, k): return f_vals[i, j, k]

    interp_val = 0.0
    for a in range(2):
        wa = (1 - dy) if a == 0 else dy
        for b in range(2):
            wb = (1 - dx1) if b == 0 else dx1
            for c in range(2):
                wc = (1 - dx2) if c == 0 else dx2
                interp_val += wa * wb * wc * val(iy + a, ix1 + b, ix2 + c)

    return interp_val

def estimate_decay_exponents(C,m):
   
    N = C.shape[0]
    lambdas = np.zeros(N)
    cut=-3.

    for k in range(N):
        moy=0
        
        row_minus = np.log(np.abs(C[k, k::  ]))
        len_minus = np.argmax(row_minus < cut)
        row_minus = row_minus[0:len_minus]
        
        if len(row_minus)>2:
            y = np.asarray(row_minus)
            x = np.arange(len(y))*(1-m)
            slope_minus, intercept_minus = np.polyfit(x,y, 1)
            moy+=1
            lambdas[k]-=slope_minus**(-1)

        
        row_plus  = np.log(np.abs(C[k, k::-1]))        
        len_plus  = np.argmax(row_plus < cut)
        row_plus  = row_plus[0:len_plus]
        
        if len(row_plus)>2:
            y = np.asarray(row_plus)
            x = np.arange(len(y))*(1-m)
            slope_plus, intercept_plus = np.polyfit(x,y, 1)
            moy+=1
            lambdas[k]-=slope_plus**(-1)
        
        lambdas[k]=lambdas[k]/moy


    return lambdas

def power_law_fit(f_y,f_x):
    guess=(max(f_x)+0.1,0.6)
    cut=0.003
    def sol(guess):
        α_c=max(abs(guess[0]),max(f_x)+cut)
        λ  =abs(guess[1])
        
        f_test= lambda x: (α_c-x)**(-λ)
        f_y_test=f_test(f_x)
        
        return np.average((abs(f_y-f_y_test)[int((1/2)*len(f_x)):len(f_x)-2])**3)
    
    result=optimize.fmin(sol,guess,disp=False)
    α_c=max(abs(result[0]),max(f_x)+cut)
    λ  =abs(result[1])
    
    f_y_power_law_func= lambda x: (α_c-x)**(-λ)
    
    
    return α_c,λ,f_y_power_law_func(f_x)
    
        
    
#####################
#####################


##### Coarsed-grained potential ######
######################################

def Generating_Chain_matrix(κ_plus,κ_minus,m,M,N_eigvec,N_discr):
    m_=m/M
    N_=M-m*m/M
    
    ######## Discretization 
    P_w_κ=np.linspace(κ_minus,κ_plus,N_discr)
    Δ_w_κ=np.append((np.roll(P_w_κ,-1)-P_w_κ)[0:len(P_w_κ)-1],P_w_κ[1]-P_w_κ[0])
    ########
    
    def Matrix_creation(P_w,Δ_w):
    
        print("Mat (loss): original",N_discr)
        @numba.jit(nopython=True, fastmath=True)
        def Mat0_simple():

            ##### Tranfer_1(w_new,w_old) = int_(w_old_)^(w_old) dw transfer(w_new,w)  !!! allows to go to smaller values of 1-m    
            transfer1_= lambda w_new,w_old       : np.exp(-(w_new-m_*w_old)*(w_new-m_*w_old)/(2*N_))/np.sqrt(2*pi*N_)

            Mat0=np.zeros((N_discr,N_discr))
            for i in range(N_discr):
                for j in range(N_discr):
                    Mat0[i,j] = transfer1_(P_w[j],P_w[i])*Δ_w[i]

            return Mat0
            
        Mat0=Mat0_simple()
        return Mat0
    
    Mat_loss=Matrix_creation(P_w_κ,Δ_w_κ)
    
    print('Diag.')
    eigenvalue, eigenvector=diag_mat(Mat_loss,N_eigvec)
    print('Fin')

    return eigenvector,P_w_κ

######################################
######################################






##### Energies #####
####################

#######  W  #########
@numba.jit(nopython=True, fastmath=True)   
def Field_w_initialization(C_inv_mat,W,N):
    Field_W_new=np.zeros(N)

    for i in range(N):
        index=i
        
        for j in range(0,index):
            Field_W_new[i]+=W[j]*C_inv_mat[j,index]
        for j in range(index+1,N):
            Field_W_new[i]+=W[j]*C_inv_mat[j,index]

    return Field_W_new

@numba.jit(nopython=True, fastmath=True)   
def Field_w_update(C_inv_mat, Field_W,w_old,w_new,index,N):

    for i in range(0,index):
        Field_W[i]+=(w_new-w_old)*C_inv_mat[i,index]
    for i in range(index+1,N):
        Field_W[i]+=(w_new-w_old)*C_inv_mat[i,index]

@numba.jit(nopython=True, fastmath=True)   
def ΔE_nn(Pot_nn_grid,m_nn_grid,W_grid,  W,w_old,w_new,index,N,C_mat):
    ΔE=0
    if index!=0:

        ΔE+=+interp3d_numba(C_mat[index-1,index], W[index-1], w_new     , m_nn_grid, W_grid, Pot_nn_grid)\
            -interp3d_numba(C_mat[index-1,index], W[index-1], w_old     , m_nn_grid, W_grid, Pot_nn_grid)
            
    if index!=N-1:

        ΔE+=+interp3d_numba(C_mat[index,index+1], w_new     , W[index+1], m_nn_grid, W_grid, Pot_nn_grid)\
            -interp3d_numba(C_mat[index,index+1], w_old     , W[index+1], m_nn_grid, W_grid, Pot_nn_grid)
            
    return ΔE
        
@numba.jit(nopython=True, fastmath=True)   
def Biased_W_sampling(Pot_nn_grid,m_nn_grid,W_grid,   W_max,W_min,W,C_inv,C_mat,Field_w,index):
    N_sample=100
    P_x=np.linspace(W_min,W_max,N_sample)
    P_y=np.zeros(N_sample)
    
    erf_ =math.erf((W_min*C_inv[index,index]+Field_w[index])/np.sqrt(2*C_inv[index,index]))
    Norm =math.erf((W_max*C_inv[index,index]+Field_w[index])/np.sqrt(2*C_inv[index,index]))-erf_
    
    for k in range(N_sample):
        P_y[k]=(math.erf((P_x[k]*C_inv[index,index]+Field_w[index])/np.sqrt(2*C_inv[index,index]))-erf_)/Norm
        

    a=np.random.rand()
    W_new=np.interp(a,P_y,P_x)
    
    b=np.random.rand()
    ΔE=ΔE_nn(Pot_nn_grid,m_nn_grid,W_grid,  W,W[index],W_new,index,N,C_mat)
    
    if np.exp(-ΔE)>b:
        Field_w_update(C_inv, Field_w,W[index],W_new,index,N)
        W[index]=W_new
    
####################
####################
def Fields_dynamics(α,κ,N,T,mo,No, C_mat_0): 
    No=int(T/(N*(1-mo)))
    
    ### Initialization of fields ###
    W0    = np.zeros(N)
    C_mat = C_mat_0
        

    N_W=400
    N_m_nn=50
     
    Pot_nn_grid=np.zeros((N_m_nn,N_W,N_W))
    Pot_nn_grid_sym=np.zeros((N_m_nn,N_W,N_W))
        
        
    W_grid=np.zeros(N_W)
    m_nn_grid=np.zeros(N_m_nn)
        
    κ_plus=κ
    κ_minus=-κ
        
    m_nn_minus=mo**No
    m_nn_plus=mo
        
    file=open('Pot_nn_kappa_'+str(κ)+'_'+str(T)+'_'+str(N)+'.txt','r')  
    line=file.readline()
        
    for i in range(N_m_nn):
        
            m_nn_grid[i]=m_nn_minus+(m_nn_plus-m_nn_minus)*(i/N_m_nn)
            W_grid=np.linspace(κ_minus,κ_plus,N_W)
            for j in range(N_W):
                for k in range(N_W):
                    
                    line=file.readline()
                    line_=line.split()
                    Pot_nn_grid[i,j,k]=float(line_[3])
     
    for i in range(N_m_nn):
            for j in range(N_W):
                for k in range(N_W):
                        
                    Pot_nn_grid_sym[i,j,k]=+(Pot_nn_grid[i,j,k]            +Pot_nn_grid[i,k,j]            )/4  \
                                           +(Pot_nn_grid[i,N_W-1-j,N_W-1-k]+Pot_nn_grid[i,N_W-1-k,N_W-1-j])/4
                                           
    file.close()  
                    
    ##### MC of the margin contribution #####
    Direct_Dynamics_ene(N,No,α,κ,mo,  C_mat,   Pot_nn_grid_sym,m_nn_grid,W_grid,   W0)
        

def Direct_Dynamics_ene(N,No,α,κ,mo,  C_mat,  Pot_nn_grid,m_nn_grid,W_grid,   W0):   
    print('No mem distrib')
    N_eigvec=1
    N_discr=max(int(250*(2*κ)/np.sqrt(1-mo**2)),1000)
    eigenvector,P_w_κ=Generating_Chain_matrix(κ,-κ,mo,1,N_eigvec,N_discr)
    
    func_1  =lambda w: np.exp(-w*w/2)*np.interp(w,P_w_κ,(eigenvector[:,0]))
    func_2  =lambda w: np.exp(-w*w/2)*np.interp(w,P_w_κ,(eigenvector[:,0])**2)
    func_typ= lambda w: np.exp(-w*w/2)
    
    P_w1=func_1(P_w_κ)/(integrate.quad(func_1,-κ,κ)[0])
    P_w2=func_2(P_w_κ)/(integrate.quad(func_2,-κ,κ)[0])
    Norm=integrate.quad(func_typ,-κ,κ)[0]
    P_w_typ=func_typ(P_w_κ)/Norm
    P_w_typ[0]=0
    P_w_typ[len(P_w_κ)-1]=0
    
    
    
    
    print("Start MC sampling")
    C_inv_mat=linalg.inv(C_mat)
    
    ### Initialization of spins ###
    W                = W0
    W_storage        = W0
    Field_W          = Field_w_initialization(C_inv_mat,W,N)
    
    ### Param. for the MC ###
    t_max=3500000*N
    t_mem_update=5*N
    
    Stop=0
    t=0
    N_storage=1
    while Stop==0:
        t+=1
        
        i = np.random.randint(N)                                                              # Select a random margin to change
        Biased_W_sampling(Pot_nn_grid,m_nn_grid,W_grid,   κ,-κ,W,C_inv_mat,C_mat,Field_W,i)   # Update themargins and their field
        
        ### Update the memory ###
        if t%t_mem_update==0:
            
            N_storage+=1
            W_storage=np.vstack((W_storage,W))
            
            if t>t_max:
                Stop=1

        if t%(5000*t_mem_update)==0 and N_storage>300:
            plt.hist(W_storage[:,0]       , bins=300, density=True,alpha=0.9,label=r'$P^{\rm edge}(w)$')
            plt.hist(W_storage[:,int(N/2)], bins=300, density=True,alpha=0.5,label=r'$P^{\rm core}(w)$')
            
            plt.plot(P_w_κ,P_w_typ            ,c='black',label=r'$P^{\rm typ.}(w)$')
            plt.plot(P_w_κ,P_w1,linestyle='--',c='black',label=r'$P_{\rm no-mem.}^{\rm edge}(w)$')
            plt.plot(P_w_κ,P_w2,linestyle='-.',c='black',label=r'$P_{\rm no-mem.}^{\rm core}(w)$')

            plt.xlabel(r'$w^\mu=\frac{\xi^\mu\cdot{\bf x}}{\sqrt{N}}$')
            plt.ylabel(r'$P\left(w\right)$')
            plt.legend()
            plt.tight_layout()
            plt.savefig('Margin_distrib_kappa_'+str(κ)+'_alpha_'+str(α)+'.pdf')
            plt.show()
            
    return  W_storage




N=200
##### Correlation functions ######
lin_correlation=1
if lin_correlation==1:
    
    κ_list=[0.5,0.75,1.0,1.25]

    for p in range(4):
        κ=κ_list[p]
        if p==0:
            α_list=np.linspace(0.225,0.45,10)
        if p==1:
            α_list=np.linspace(0.325,0.7,16)
        if p==2:
            α_list=np.linspace(0.325,0.975,27)    
        if p==3:
            α_list0=np.linspace(0.325,0.575,6) 
            α_list1=np.linspace(0.625,0.85,10)    
            α_list2=np.linspace(0.89,1.29,11)   
            α_list=np.append(α_list0,α_list1)
            α_list=np.append(α_list,α_list2)
        
        c_list=(α_list-np.min(α_list))/(np.max(α_list)-np.min(α_list))
        index_list=np.linspace(0, N-1,N)
        C_mat=np.zeros((len(α_list),N,N))
        C0_mat=np.zeros((N,N))
    
        fig, ax = plt.subplots(constrained_layout=True)

        for k in range(len(α_list)):
            if p==0:
                file1=open('Correlation_high_res_zoom(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')
            else:
                file1=open('Correlation_high_res(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')
            line=file1.readline()
            line=line.split()
    
            a0=line[0]
            T =int(a0[2:])
            a1=line[1]
            N =int(a1[2:])
            a2=line[2]
            m=float(a2[3:])
            a3=line[3]
            No=int(a3[3:])
    
            mo=m**No
    
            for i in range(N):
                line=file1.readline()
                line=line.split()
                C_mat[k,i,:]=line
        

                if k==0:
                    C0_mat[i,:]=mo**abs(i-index_list)

            file1.close()
        
        ax.plot([0,0],[0,0],c='black',label=r"${\bf Q}^{*}_{k,k'}$")
        ax.plot(index_list*(1-m)*No,((C0_mat[int(N/2),:])),c='black',linestyle='--',label=r"${\bf Q}^{\rm no-mem.}_{(N_0+1)k,(N_0+1)k'}$")   
        for k in range(len(α_list)):
            ax.plot(index_list*(1-m)*No,C_mat [k,int(N/2),:], color=plt.cm.viridis(c_list[k]))
        
        # ---- COLORBAR ----
        norm = colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])  # required for older matplotlib versions

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r'$\alpha$')

        # 4 ticks
        cbar.set_ticks([0.0, 0.33, 0.66, 1.0])
        index1=np.min(α_list)
        index2=np.min(α_list)+(1/3)*(np.max(α_list)-np.min(α_list))
        index3=np.min(α_list)+(2/3)*(np.max(α_list)-np.min(α_list))
        index4=np.min(α_list)+(3/3)*(np.max(α_list)-np.min(α_list))
        cbar.set_ticklabels([str(round(index1,3)), str(round(index2,3)), str(round(index3,3)), str(round(index4,3))])


        ax.text(0.79*max(index_list*(1-m)*No),0.75,r"$k'=k_f^*/2$",fontsize=9,va='center',ha='left')

        plt.xlabel(r"$k[(1-m)N_0]$")
        plt.ylabel(r"${\bf Q}^*_{k,k'}$")    
        plt.legend()
        plt.savefig('Correlation_kappa_'+str(κ)+'.pdf')
        plt.show()


##################################
##################################




##### Correlation functions (log scale)######
log_lin_correlation=0
if log_lin_correlation==1:

    κ_list=[0.5,0.75,1.0,1.25]
    N=200

    for p in range(4):
        κ=κ_list[p]
        if p==0:
            α_list=np.linspace(0.225,0.45,10)
        if p==1:
            α_list=np.linspace(0.325,0.7,16)
        if p==2:
            α_list=np.linspace(0.325,0.975,27)    
        if p==3:
            α_list0=np.linspace(0.325,0.575,6) 
            α_list1=np.linspace(0.625,0.85,10)    
            α_list2=np.linspace(0.89,1.29,11)   
            α_list=np.append(α_list0,α_list1)
            α_list=np.append(α_list,α_list2)
        
        
        c_list=(α_list-np.min(α_list))/(np.max(α_list)-np.min(α_list))
        index_list=np.linspace(0, N-1,N)
        C_mat=np.zeros((len(α_list),N,N))
        C0_mat=np.zeros((N,N))
        
        fig, ax = plt.subplots(constrained_layout=True)

        for k in range(len(α_list)):
            if p==0:
                file1=open('Correlation_high_res_zoom(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')
            else:
                file1=open('Correlation_high_res(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')        
                
            line=file1.readline()
            line=line.split()
    
            a0=line[0]
            T =int(a0[2:])
            a1=line[1]
            N =int(a1[2:])
            a2=line[2]
            m=float(a2[3:])
            a3=line[3]
            No=int(a3[3:])
    
            mo=m**No
    
            for i in range(N):
                line=file1.readline()
                line=line.split()
                C_mat[k,i,:]=line
        

                if k==0:
                    C0_mat[i,:]=mo**abs(i-index_list)

            file1.close()
        
        
        ax.plot([0,0],[0,0],c='black',label=r"${\bf Q}^{*}_{k,k'}$")
        ax.plot(index_list*(1-m)*No,np.log(np.abs(C0_mat[int(N/2),:])),c='black',linestyle='--',label=r"${\bf Q}^{\rm no-mem.}_{k,k'}$")   
        for k in range(len(α_list)):
            ax.plot(index_list*(1-m)*No,np.log(np.abs(C_mat [k,int(N/2),:])),color=plt.cm.viridis(c_list[k]))
        
        # ---- COLORBAR ----
        norm = colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])  # required for older matplotlib versions

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r'$\alpha$')

        # 4 ticks
        cbar.set_ticks([0.0, 0.33, 0.66, 1.0])
        index1=np.min(α_list)
        index2=np.min(α_list)+(1/3)*(np.max(α_list)-np.min(α_list))
        index3=np.min(α_list)+(2/3)*(np.max(α_list)-np.min(α_list))
        index4=np.min(α_list)+(3/3)*(np.max(α_list)-np.min(α_list))
        cbar.set_ticklabels([str(round(index1,3)), str(round(index2,3)), str(round(index3,3)), str(round(index4,3))])

        plt.xlabel(r"$k[(1-m)No]$")
        plt.ylabel(r"${\bf Q}^*_{k,k'}$")    
        plt.ylim([-4,0.1])
        plt.legend()
        plt.show()


##################################
##################################



##### Correlation functions (log scale) averaged######
log_lin_correlation_averaged=0
if log_lin_correlation_averaged==1:

    κ_list=[0.5,0.75,1.0,1.25]
    N=200

    for p in range(4):
        κ=κ_list[p]
        if p==0:
            α_list=np.linspace(0.225,0.45,10)
        if p==1:
            α_list=np.linspace(0.325,0.7,16)
        if p==2:
            α_list=np.linspace(0.325,0.975,27)    
        if p==3:
            α_list0=np.linspace(0.325,0.575,6) 
            α_list1=np.linspace(0.625,0.85,10)    
            α_list2=np.linspace(0.89,1.29,11)   
            α_list=np.append(α_list0,α_list1)
            α_list=np.append(α_list,α_list2)
        
        
        c_list=(α_list-np.min(α_list))/(np.max(α_list)-np.min(α_list))
        index_list=np.linspace(0, N-1,N)
        C_mat=np.zeros((len(α_list),N,N))
        C_avg=np.zeros((len(α_list),N))
        C0_mat=np.zeros((N,N))
        
        fig, ax = plt.subplots(constrained_layout=True)

        for k in range(len(α_list)):
            if p==0:
                file1=open('Correlation_high_res_zoom(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')
            else:
                file1=open('Correlation_high_res(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')        
                
            line=file1.readline()
            line=line.split()
    
            a0=line[0]
            T =int(a0[2:])
            a1=line[1]
            N =int(a1[2:])
            a2=line[2]
            m=float(a2[3:])
            a3=line[3]
            No=int(a3[3:])
    
            mo=m**No
    
            for i in range(N):
                line=file1.readline()
                line=line.split()
                C_mat[k,i,:]=line
                
                if k==0:
                    C0_mat[i,:]=mo**abs(i-index_list)

            file1.close()
            
            def Correlation_averaging(C_in):
                index=(np.linspace(1,N,N))[::-1]
                C_avg=np.zeros(N)
                for i in range(N):
                    C_in_cut = C_in[k,i,i::]
                    C_in_ext = np.pad(C_in_cut, (0, N - len(C_in_cut)), mode='constant')
                    C_avg+=C_in_ext
                C_avg=C_avg/index
                return C_avg
                
            
            C_avg[k,:]=Correlation_averaging(C_mat)  
            


        
        
        ax.plot([0,0],[0,0],c='black',label=r"${\bf Q}^{*,{\rm avg}}_{k,k'}$")
        ax.plot(index_list*(1-mo),np.log(np.abs(C0_mat[0,:])),c='black',linestyle='--',label=r"${\bf Q}^{\rm no-mem.}_{(N_0+1)k,(N_0+1)k'}$")   
        for k in range(len(α_list)):
            ax.plot(index_list*(1-mo),np.log(np.abs(C_avg [k,:])),color=plt.cm.viridis(c_list[k]))
        
        # ---- COLORBAR ----
        norm = colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])  # required for older matplotlib versions

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r'$\alpha$')

        # 4 ticks
        cbar.set_ticks([0.0, 0.33, 0.66, 1.0])
        index1=np.min(α_list)
        index2=np.min(α_list)+(1/3)*(np.max(α_list)-np.min(α_list))
        index3=np.min(α_list)+(2/3)*(np.max(α_list)-np.min(α_list))
        index4=np.min(α_list)+(3/3)*(np.max(α_list)-np.min(α_list))
        cbar.set_ticklabels([str(round(index1,3)), str(round(index2,3)), str(round(index3,3)), str(round(index4,3))])


        ind_max=np.argmin(abs(-np.log(np.abs(C_avg [len(α_list)-1,:]))-4))

        plt.xlabel(r"$(k-k')[(1-m)N_0]$")
        plt.ylabel(r"$\log\left({\bf Q}^{*,{\rm avg}}_{k,k'}\right)$")    
        plt.xlim([-0.1,1.01*ind_max*(1-m)*No])
        plt.ylim([-4,0.1])
        plt.legend()
        plt.savefig('Log_correlation_kappa_'+str(κ)+'.pdf')
        plt.show()


##################################
##################################


##### Correlation lengths #####
corr_length=0
if corr_length==1:
    κ_list=[0.5,0.75,1.0,1.25]
    linestyle_list=['dashed','dotted','dashdot','solid']

    fig, ax = plt.subplots(constrained_layout=True)
    for p in range(4):
        κ=κ_list[p]
        if p==0:
            α_list=np.linspace(0.225,0.45,10)
        if p==1:
            α_list=np.linspace(0.325,0.7,16)
        if p==2:
            α_list=np.linspace(0.325,0.975,27)    
        if p==3:
            α_list0=np.linspace(0.325,0.575,6) 
            α_list1=np.linspace(0.625,0.85,10)    
            α_list2=np.linspace(0.89,1.29,11)   
            α_list=np.append(α_list0,α_list1)
            α_list=np.append(α_list,α_list2)
        
        
        c_list=(α_list-np.min(α_list))/(np.max(α_list)-np.min(α_list))
        ξ=np.zeros((len(α_list),N))
    
        for k in range(len(α_list)):
            if p==0:
                file1=open('Correlation_high_res_zoom(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')
            else:
                file1=open('Correlation_high_res(kappa='+str(κ)+'_alpha='+str(round(α_list[k],3))+').txt')     
            line=file1.readline()
            line=line.split()
    
            a0=line[0]
            T =int(a0[2:])
            a1=line[1]
            N =int(a1[2:])
            a2=line[2]
            m=float(a2[3:])
            a3=line[3]
            No=int(a3[3:])
    
            mo=m**No
    
            C_mat=np.zeros((N,N))
    
            for i in range(N):
                line=file1.readline()
                line=line.split()
                C_mat[i,:]=line
        

            ξ[k,:]=estimate_decay_exponents(C_mat,mo)
            file1.close()
     
        ξ_moy=np.average(ξ,axis=1)
        α_c,λ,ξ_power_law=power_law_fit(ξ_moy,α_list)
        print(α_c,λ)
        
        ax.plot(np.append([0.2],α_list[:]),np.append([1],ξ[:,0])       ,c='blue')
        ax.plot(np.append([0.2],α_list[:]),np.append([1],ξ[:,int(N/4)]),c='orange')
        ax.plot(np.append([0.2],α_list[:]),np.append([1],ξ[:,int(N/2)]),c='green' )
        ax.plot(np.append([0.2],α_list[:]),np.append([1],ξ[:,int(3*N/4)]),c='red',alpha=0.7)
        ax.plot(np.append([0.2],α_list[:]),np.append([1],ξ[:,N-1]),c='grey' ,alpha=0.7)
        
        
    
        ξ_max=max(max(ξ[:,0]),max(ξ[:,int(N/4)]),max(ξ[:,int(N/2)]),max(ξ[:,int(3*N/4)]),max(ξ[:,(N-1)]))
        
        ax.text(max(α_list)-0.05,ξ_max+0.4,r'$\kappa=$'+str(κ),fontsize=11,va='center',ha='left')
        
   
    
    ax.plot([min(α_list),min(α_list)],[min(ξ[:,0]),min(ξ[:,0])],c='blue',label=r"${\xi}_{k=0}$")
    ax.plot([min(α_list),min(α_list)],[min(ξ[:,0]),min(ξ[:,0])],c='orange',label=r"${\xi}_{k={k_f^*}/{4}}$")
    ax.plot([min(α_list),min(α_list)],[min(ξ[:,0]),min(ξ[:,0])],c='green',label=r"${\xi}_{k={k_f^*}/{2}}$")
    ax.plot([min(α_list),min(α_list)],[min(ξ[:,0]),min(ξ[:,0])],c='red',label=r"${\xi}_{k={3k_f^*}/{4}}$")
    ax.plot([min(α_list),min(α_list)],[min(ξ[:,0]),min(ξ[:,0])],c='grey',label=r"${\xi}_{k={k_f^*}}$")
    ax.plot([0.2,1.4],[1,1],c='black',linestyle='--')
    ax.text(1.2,1.25,r'$\xi_{\rm no-mem.}$')
   
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"${\xi}_k$")
    plt.xlim([0.2,1.4])
    plt.ylim([0.5,8])
    plt.legend(loc="upper left")
    plt.savefig('Correlation_length.pdf')
    plt.show()

##################################
##################################



##### Phase diagram #####
phase_diagram=0
if  phase_diagram==1:
    κ_no_mem=np.array([0.4,  0.6, 0.8, 1.0])
    α_no_mem=np.array([0.195,0.36,0.54,0.74])
    
    κ_c=np.array([0.35 ,0.5,0.75,1.0,1.25])
    α_c=np.array([0.3,0.45,0.7,0.975,1.29])
    

    κ_smooth_no_mem = np.linspace(κ_no_mem.min(), κ_c.max(), 300)
    coeffs = np.polyfit(κ_no_mem,α_no_mem, deg=4)
    p_no_mem = np.poly1d(coeffs)
    α_smooth_no_mem = p_no_mem(κ_smooth_no_mem)
        
    κ_smooth_c = np.linspace(κ_c.min(), κ_c.max(), 300)
    coeffs = np.polyfit(κ_c,α_c, deg=4)
    p_c = np.poly1d(coeffs)
    α_smooth_c = p_c(κ_smooth_c)




    N_SAT=500
    κ_SAT=np.linspace(0,0.814,N_SAT)
    α_SAT=np.zeros(N_SAT)
    for k in range(N_SAT):
        if κ_SAT[k]==0:
            α_SAT[k]=0.01
        else: 
            H=np.log(math.erf(κ_SAT[k]/np.sqrt(2)))
            α_SAT[k]=-np.log(2)/H
        
    N_OGP=400
    α_OGP=np.linspace(1.29,0.0101,N_OGP)
    κ_OGP=np.zeros(N_OGP)
    m_OGP=np.zeros(N_OGP)
    guess=[0.97,0.909]
    guess_RS=[0.97,np.arctanh(0.97),0.97*0.97,0,0.155]
    
    for k in range(N_OGP):
        if α_OGP[k]==0:
            κ_OGP[k]=0
        else: 
            κ_0=np.interp(α_OGP[k],α_SAT,κ_SAT)
            def OGP_annealed_free_energy(x):
                m     = max(0.05,min(abs(x[0]),0.99999))
                m_hat = np.arctanh(m)
                κ     = max(abs(x[1]),κ_0)
                
                s_1=-m*m_hat+np.log(2*np.cosh(m_hat))
                
                h= lambda w:   special.erf((κ-m*w)/np.sqrt(2*(1-m*m)))/2+special.erf((κ+m*w)/np.sqrt(2*(1-m*m)))/2
                f= lambda w : np.exp(-w*w/2)*np.log(h(w))
                g= lambda w : np.exp(-w*w/2)  
                s_2=integrate.quad(f,-κ_0,+κ_0)[0]/integrate.quad(g,-κ_0,κ_0)[0]
                return s_1+α_OGP[k]*s_2
            
            def OGP_RS_free_energy(x):
                m     = min(abs(x[0]),0.99)
                m_hat = abs(x[1])
                q     = max(abs(x[2]),m*m+0.001)
                q_hat = abs(x[3])
                
                κ     = max(abs(x[4]),κ_0)
                
                f_entr= lambda z:(np.exp(-z*z/2)/np.sqrt(2*pi))*np.log(2*np.cosh(np.sqrt(q_hat)*z+m_hat))
                s_1=-m*m_hat#-(1-q)*q_hat/2#+integrate.quad(f_entr,-8,8)[0]
                print('<sgs<',m,m_hat,q,q_hat)
                
                
                
                h_energ= lambda w,z:   special.erf((κ-np.sqrt(q-m*m)*z-m*w)/np.sqrt(2*(1-q)))/2+special.erf((κ+np.sqrt(q-m*m)*z+m*w)/np.sqrt(2*(1-q)))/2
                f_energ= lambda w,z : (np.exp(-w*w/2-z*z/2)/np.sqrt(2*pi))*np.log(h_energ(w,z))
                f_norm= lambda w : np.exp(-w*w/2)  
                s_2=integrate.dblquad(f_energ,-8,8,-κ_0,+κ_0)[0]/integrate.quad(f_norm,-κ_0,κ_0)[0]
                return s_1#+α_OGP[k]*s_2
            
            def sol(guess):
                d=0.001
                a=OGP_annealed_free_energy(guess)
                b=OGP_annealed_free_energy([guess[0]+d,guess[1]]) 
                return [a,(b-a)/d]
            
            def sol_RS(guess):
                d=0.001
                a=OGP_RS_free_energy(guess)
                b=OGP_RS_free_energy([guess[0]+d,guess[1]  ,guess[2]  ,guess[3]  ,guess[4]])
                c=OGP_RS_free_energy([guess[0]  ,guess[1]+d,guess[2]  ,guess[3]  ,guess[4]])
                e=OGP_RS_free_energy([guess[0]  ,guess[1]  ,guess[2]+d,guess[3]  ,guess[4]])
                f=OGP_RS_free_energy([guess[0]  ,guess[1]  ,guess[2]  ,guess[3]+d,guess[4]])
                
                eq1=a
                eq2=(b-a)/d
                eq3=(c-a)/d
                eq4=(e-a)/d
                eq5=(f-a)/d
                print(guess)
                print(eq1,eq2,eq3,eq4,eq5)
                print('')
                #return eq1,eq2,eq3,eq4,eq5
                return 30*(abs(eq1))**(1/2)+np.sqrt(abs(eq2))+np.sqrt(abs(eq3))+np.sqrt(abs(eq4))+np.sqrt(abs(eq5))
            
        
            guess=optimize.fsolve(sol,np.abs(guess))
            κ_OGP[k]=abs(guess[1])
            m_OGP[k]=abs(guess[0])

           # guess_RS=optimize.fmin(sol_RS,(guess_RS))
           # κ_OGP[k]=abs(guess_RS[4])


    fig, ax = plt.subplots(constrained_layout=True)
    
    
    ## No-mem ##
    ax.plot(α_smooth_no_mem,κ_smooth_no_mem,linestyle='-.',c='red',zorder=1)
    ax.scatter(α_no_mem,κ_no_mem,c='black',s=20,zorder=2)
    ax.scatter(α_no_mem,κ_no_mem,c='red',s=8,zorder=3)
    ax.text(1.05*max(α_no_mem),1.13*max(κ_no_mem),
            "no-memory\n instability",
            bbox=dict(boxstyle="round",facecolor="white",edgecolor="red",alpha=1),
            fontsize=8,va='center',ha='left',zorder=7)
    ax.text(0.3*max(α_c),0.8*max(κ_c),
            "EASY",
            bbox=dict(boxstyle="round",facecolor="white",edgecolor="green",alpha=1),
            fontsize=11,va='center',ha='center',zorder=7)
    ####

    ## Connected ##
    ax.plot(α_smooth_c,κ_smooth_c,linestyle='-.',c='orange',zorder=4)
    ax.scatter(α_c,κ_c,c='black',s=20,zorder=5)
    ax.scatter(α_c,κ_c,c='orange',s=8,zorder=6)
    ax.text(0.37*max(α_c),0.5*max(κ_c),
            "connected\n transition",
            bbox=dict(boxstyle="round",facecolor="white",edgecolor="orange",alpha=1),
            fontsize=8,va='center',ha='left',zorder=7)
    ax.text(0.75*max(α_c),0.7*max(κ_c),
            "HARD",
            bbox=dict(boxstyle="round",facecolor="white",edgecolor="orange",alpha=1),
            fontsize=11,va='center',ha='left',zorder=12)
    ####

    ## SAT ##
    ax.plot(α_SAT,κ_SAT,c='blue',zorder=7)
    ax.text(0.6*max(α_SAT),0.3*max(κ_SAT),
            "UNSAT",
            bbox=dict(boxstyle="round",facecolor="white",edgecolor="blue",alpha=1),
            fontsize=11,va='center',ha='left',zorder=8)
    ####
    
    ## OGP ##
    ax.plot(np.append(α_OGP,[0]),np.append(κ_OGP,[0]),c='black',zorder=8,linestyle='--')
    ax.text(0.1*max(α_OGP),0.2*max(κ_OGP),
            "OGP\n(annealed)",
            bbox=dict(boxstyle="round",facecolor="white",edgecolor="black",alpha=1),
            fontsize=8,va='center',ha='center',zorder=9)
    ####

  
    ### filling areas ###
    
    κ_c_2=np.array([0.15,0.35 ,0.5,0.75,1.0,1.25])
    α_c_2=np.array([0.1,0.3,0.45,0.7,0.975,1.29])
        
    κ_smooth_c_2 = np.linspace(κ_c_2.min(), κ_c_2.max(), 300)
    coeffs = np.polyfit(κ_c_2,α_c_2, deg=4)
    p_c_2 = np.poly1d(coeffs)
    α_smooth_c_2 = p_c_2(κ_smooth_c_2)
    
    x            = p_c_2(κ_smooth_c_2)
    x_fade_start = α_c_2[1]
    
    y_up = 1.3*np.ones(len(x))
    y_c  = κ_smooth_c_2
    y_SAT= np.interp(x,α_SAT,κ_SAT)
    
    # Build alpha profile
    alpha_max=0.2
    alpha = alpha_max*np.ones_like(x)
    mask = x <= x_fade_start

    alpha[mask] = np.linspace(0,alpha_max, mask.sum())


    # Make 2D gradient for imshow
    grad = np.vstack([alpha, alpha])

    # Draw gradient image
    rgba = np.zeros((*grad.shape, 4))
    rgba[..., 0] = 1.0   # R
    rgba[..., 1] = 0.65   # G
    rgba[..., 2] = 0.0   # B
    rgba[..., 3] = grad  # alpha gradient

    im = ax.imshow(
        rgba,
        extent=[x.min(), x.max(),
            min(y_SAT.min(), y_c.min()),
            max(y_SAT.max(), y_c.max())],
        origin="lower",
        aspect="auto"
    )

    # Clip to area between curves
    verts = np.concatenate([
        np.column_stack([x, y_SAT]),
        np.column_stack([x[::-1], y_c[::-1]])
    ])

    path = Path(verts)
    patch = PathPatch(
    path,
    facecolor="none",
    edgecolor="none",
    linewidth=0
)
    ax.add_patch(patch)
    im.set_clip_path(patch)



    rgba = np.zeros((*grad.shape, 4))
    rgba[..., 0] = 0.0   # R
    rgba[..., 1] = 0.8   # G
    rgba[..., 2] = 0.0   # B
    rgba[..., 3] = grad  # alpha gradient

    im = ax.imshow(
        rgba,
        extent=[x.min(), x.max(),
            min(y_c.min(), y_up.min()),
            max(y_c.max(), y_up.max())],
        origin="lower",
        aspect="auto"
    )

    # Clip to area between curves
    verts = np.concatenate([
        np.column_stack([x, y_c]),
        np.column_stack([x[::-1], y_up[::-1]])
    ])

    path = Path(verts)
    patch = PathPatch(
    path,
    facecolor="none",
    edgecolor="none",
    linewidth=0
)
    ax.add_patch(patch)
    im.set_clip_path(patch)
                
        
    ax.fill_between(α_SAT, np.zeros(len(α_SAT)), κ_SAT, where=(np.zeros(len(α_SAT)) < κ_SAT), color='blue', alpha=0.1,
                 interpolate=True)
        
    plt.xlim([-0.02,1.32])
    plt.ylim([-0.04,1.34])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\kappa$')
    plt.savefig('Phase_diagram.pdf')
    plt.show()
    



##################################
##################################


##### Margins distribution #####
margin=0
read=0
if margin==1 and read==0:
    T_tot=70
    N=200
    mo=0.9995
    No=int(T_tot/(N*(1-mo)))

    κ  = 1.0
    α  = 0.85

    file1=open('Correlation_high_res(kappa='+str(κ)+'_alpha='+str(α)+').txt')
    line=file1.readline()
    line=line.split()

    C_mat=np.zeros((N,N))
    for i in range(N):
        line=file1.readline()
        line=line.split()
        C_mat[i,:]=line
        
    file1.close()
    W_store=Fields_dynamics(α,κ,N,T_tot,mo,No, C_mat)
    
    
    file1=open('Margin_distrib(kappa='+str(κ)+'_alpha='+str(α)+').txt','w')
    for k in range(len(W_store[:,0])):
        file1.write(W_store[k,:])
        file1.write('  ')
    file1.close()    
    




   # T_tot=70
   # N=200
   # mo=0.9995
   # No=int(T_tot/(N*(1-mo)))

   # κ  = 1.0
   # α  = 0.4

   # file1=open('Correlation_high_res(kappa='+str(κ)+'_alpha='+str(α)+').txt')
   # line=file1.readline()
   # line=line.split()

   # C_mat=np.zeros((N,N))
   # for i in range(N):
   #     line=file1.readline()
   #     line=line.split()
   #     C_mat[i,:]=line
        
   # file1.close()
   # W_store=Fields_dynamics(α,κ,N,T_tot,mo,No, C_mat)
    
    
   # file1=open('Margin_distrib(kappa='+str(κ)+'_alpha='+str(α)+').txt','w')
   # for k in range(len(W_store)):
   #     file1.write(W_store[k])
   #     file1.write('  ')
   # file1.close()    
    







