import numpy as np

import numba 

import scipy as sc
from scipy import integrate, optimize, special, linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs,gmres
from scipy.interpolate import RegularGridInterpolator

from itertools import product

import matplotlib.pylab as plt
import math 

import random


pi=np.arccos(-1)

##### Functions #####
#####################
def diag_mat(matrix,N_eigvec):
    matrix_sparse=csr_matrix(matrix)
    eigenvalue, eigenvector = eigs(matrix_sparse, k=N_eigvec, which='LM')
    return np.real(eigenvalue), np.real(eigenvector)

@numba.jit(nopython=True, fastmath=True)
def power_mat_partial_and_rescale(eigenvalue_partial,eigenvector_partial,P_w,Δ_w,time,N_eigvec):
    n = len(P_w)
    eigenvector_inv_partial= np.linalg.pinv(eigenvector_partial)
    Mat_new = np.zeros((n, n), dtype=np.complex64)

    print('Building partial matrix power')
    print(eigenvalue_partial[:]**time)
    print('')
    for i in range(n):
        for j in range(n):
            for k in range(N_eigvec):
                Mat_new[i, j] += eigenvector_partial[i, k] * (eigenvalue_partial[k]**time) * eigenvector_inv_partial[k, j]/Δ_w[j]

    return Mat_new

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
    
        print("Mat (loss): original")
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

    return eigenvalue,eigenvector,P_w_κ,Δ_w_κ,Mat_loss

def Construct_coarsed_grained_nn_interaction(κ_plus,κ_minus,m_nn,mo,No):

    @numba.jit(nopython=True, fastmath=True)
    def Potential_eff_Markov(P_w,m,M,No):
        
        n=len(P_w)
        Pot=np.zeros((n,n))
        
        m_nn_eff=(m**No)/(M**(No-1))
        No   =M-m*m/M
        N_eff=M-m_nn_eff*m_nn_eff/M
        
        C0_eff= (m_nn_eff/M)**2/N_eff
        C1_eff= (     1    )   /N_eff
        C01_eff=-(m_nn_eff/M)/N_eff
        
        for i in range(n):
            for j in range(n):
                interactions=-P_w[j]*P_w[j]*C0_eff/2-P_w[i]*P_w[i]*C1_eff/2-P_w[i]*P_w[j]*C01_eff-np.log(2*pi*N_eff)/2
                Pot[i,j]+=interactions
                             
        return Pot
                                
    def Param_sol(x):
        Co_inv_00=x[0]
        Co_inv_01=x[1]
        C_inv_00=x[2]
        C_inv_01=x[3]
        
        def Mat_coarsed_grained(Co_00,Co_01,C_00,C_01):
        
            m=(1-Co_00/np.sqrt(Co_00**2-4*Co_01**2))/(2*Co_01)
            M=1/np.sqrt(Co_00**2-4*Co_01**2)
            m_nn_eff=(m**(No-1))/(M**(No-2))
            
            A=(m**2/M**2)/(M-m**2/M)+1/(M-m_nn_eff**2/M)
            B=-(m_nn_eff/M)/(M-m_nn_eff**2/M)
            C=C_00-1/(M-m**2/M)+1/(M-m_nn_eff**2/M)
            
            Mat=[[C_00,Co_01,C_01],[Co_01,A,B],[C_01,B,C]]
            
            return np.linalg.inv(Mat)
        
        Mat=Mat_coarsed_grained(Co_inv_00,Co_inv_01,C_inv_00,C_inv_01   )
        
        eq1=Mat[0,0]-1 
        eq2=Mat[0,1]-mo 
        eq3=Mat[0,2]-m_nn 
        eq4=Mat[1,1]-1
        
        return [eq1,eq2,eq3,eq4]
        
    sol=optimize.fsolve(Param_sol,[(1+mo*mo)/(1-mo*mo),-mo/(1-mo*mo),1/(1-mo*mo),0])
    
    ### Getting the coefficiant of the 1-loop C^{-1} correlation matrix
    Co_00=sol[0]
    Co_01=sol[1]
    C_00 =sol[2]
    C_01 =sol[3]
    
    ### Rewriting the process inside the loop as a Markov-Chain <w_t w_t>=M // <w_t w_{t+1}>=m
    M=1/np.sqrt(Co_00**2-4*Co_01**2)
    m=(1-Co_00/np.sqrt(Co_00**2-4*Co_01**2))/(2*Co_01)
    N_=M-m*m/M
    
    
    print('Start: Generating effective nearest neighbor loss')
    print('')
    N_eigvec=60
    N_discr=max(int(250*(κ_plus-κ_minus)/np.sqrt(N_)),1000)
    print('Matrix size:',N_discr,'x',N_discr,'  N eigenvectors:',N_eigvec)
    print('Params eff. Markov chain: m=',round(m,5),'  M=',round(M,5),'   N_=',N_)
    
    eigenvalue_partial,eigenvector_partial,P_w_κ,Δ_w_κ,Mat_loss=Generating_Chain_matrix(κ_plus=κ_plus,κ_minus=κ_minus,m=m,M=M,    N_eigvec=N_eigvec,N_discr=N_discr)
    Mat     = power_mat_partial_and_rescale(eigenvalue_partial,eigenvector_partial,P_w_κ,Δ_w_κ,No,N_eigvec)
    Mat_eff = Potential_eff_Markov(P_w_κ,m,M,No)
    
    
    test=0
    if test==1:
        Mat_eff_loss     =Potential_eff_Markov(P_w_κ,m,M,1)
        
        power=0
        if power==1:
            print('Pw')
            mat_pow=np.linalg.matrix_power(Mat_loss, No)
    
        N_test=30
        for k in range(N_test):
                
                index=int(N_discr*(k/N_test))
                plt.plot(P_w_κ,np.log(abs(Mat_loss[index,:]/Δ_w_κ[:])),label='OG')
                plt.plot(P_w_κ,(Mat_eff_loss[index,:]),label='OG Markov chain')
                plt.plot(P_w_κ,(Mat_eff_loss[:,index]),label='OG Markov chain2')
                plt.plot(P_w_κ,np.log(abs(Mat[index,:])),linestyle='-.',label='est. power')
                plt.plot(P_w_κ,np.log(abs(Mat[:,index])),linestyle='-.',label='est. power2')
                if power==1:
                    plt.plot(P_w_κ,np.log(abs(mat_pow[index,:]/Δ_w_κ[:])),linestyle='--',label='power')
                    plt.plot(P_w_κ,np.log(abs(mat_pow[:,index]/Δ_w_κ[:])),linestyle='--',label='power2')
                plt.plot(P_w_κ,Mat_eff[index,:],label='Markov chain')        
                plt.plot(P_w_κ,Mat_eff[:,index],label='Markov chain2')      
            
                plt.plot(P_w_κ,np.log(abs(Mat[index,:]))-Mat_eff[index,:],linestyle='--',label='est. Δpower')
                plt.plot(P_w_κ,np.log(abs(Mat[:,index]))-Mat_eff[:,index],linestyle='--',label='est. Δpower2')
                plt.legend()
                plt.ylim([-8+max(Mat_eff[index,:]),1+max(Mat_eff[index,:])])
                plt.title('mo**No: '+str(round(m**N,5))+'  N_:'+str(round(N_,5)))
                plt.show()

        N_test=30
        for k in range(N_test):
                
                index=int(N_discr*(k/N_test))
                plt.plot(P_w_κ,np.exp(np.log(abs(Mat_loss[index,:]/Δ_w_κ[:]))),label='OG')
                plt.plot(P_w_κ,np.exp((Mat_eff_loss[index,:])),label='OG Markov chain')
                plt.plot(P_w_κ,np.exp(np.log(abs(Mat[index,:]))),linestyle='-.',label='est. power')
                if power==1:
                    plt.plot(P_w_κ,np.exp(np.log(abs(mat_pow[index,:]/Δ_w_κ[:]))),linestyle='--',label='power')
                plt.plot(P_w_κ,np.exp(Mat_eff[index,:]),label='Markov chain')        
            
                plt.legend()
                plt.ylim([0,1+max(np.exp(Mat_eff[index,:]))])
                plt.title('mo**No: '+str(round(m**N,5))+'  N_:'+str(round(N_,5)))
                plt.show()

    print('End: Generating effective nearest neighbor loss')
    
    return np.log(abs(Mat))-np.transpose(Mat_eff),P_w_κ
    
def Construct_all_nn_interaction(κ_plus,κ_minus,m_nn_minus,m_nn_plus,N_m_nn,N_W,mo,No):
    Pot_nn_grid=np.zeros((N_m_nn,N_W,N_W))
    W_grid_nn=np.linspace(κ_minus,κ_plus,N_W)
    m_nn_grid=np.linspace(m_nn_minus,m_nn_plus,N_m_nn)

    
    def Create_func(Pot_nn,W_grid,Pot_interp,index):
        lenght=len(W_grid)
        
        for i in range(lenght):
            for j in range(lenght):
                Pot_nn[index,i,j]=Pot_interp(W_grid[i],W_grid[j])
                     
    
    for k in range(N_m_nn):
        print('!!!!!  k:',k+1,'  k_tot:',N_m_nn,'  kappa:'+str(κ_plus)+'  m_nn:',m_nn_grid[k])
        m_nn=m_nn_grid[k]
        Pot,W_grid=Construct_coarsed_grained_nn_interaction(κ_plus,κ_minus,m_nn,mo,No)
        
        for i in range(N_W):
            for j in range(N_W):
                Pot_nn_grid[k,i,j]=-interp2d_numba(W_grid_nn[i], W_grid_nn[j], W_grid, Pot)

    test=0
    if test==1:
        
        @numba.jit(nopython=True, fastmath=True)
        def Potential_eff_Markov(P_w,m,M,No):
            
            n=len(P_w)
            Pot=np.zeros((n,n))
            
            m_nn_eff=(m**No)/(M**(No-1))
            No   =M-m*m/M
            N_eff=M-m_nn_eff*m_nn_eff/M
            
            C0_eff= (m_nn_eff/M)**2/N_eff
            C1_eff= (     1    )   /N_eff
            C01_eff=-(m_nn_eff/M)/N_eff
            
            for i in range(n):
                for j in range(n):
                    interactions=-P_w[i]*P_w[i]*C0_eff/2-P_w[j]*P_w[j]*C1_eff/2-P_w[i]*P_w[j]*C01_eff-np.log(2*pi*N_eff)/2
                    Pot[i,j]+=interactions
                                 
            return Pot
        
        
        Pot_Markov_grid=np.zeros((N_m_nn,N_W,N_W))
        for k in range(N_m_nn):
            m_nn=m_nn_grid[k]
            Pot_Markov=Potential_eff_Markov(W_grid,m_nn,1,1)
        
            for i in range(N_W):
                for j in range(N_W):
                    Pot_Markov_grid[k,i,j]=interp2d_numba(W_grid_nn[i], W_grid_nn[j], W_grid, Pot_Markov)
        
        
        get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
        color = get_colors(N_m_nn)
        
        N_w_test=6
        N_m_test=6
        for j in range(N_w_test):
            w_test=(j/N_w_test)*κ_plus
     
            for k in range(N_m_test):
                m_nn=m_nn_minus+(m_nn_plus-m_nn_minus)*(k/N_m_test)
                
                
                Func_=[]
                for l in range(N_W):
                    Func_=np.append(Func_,interp3d_numba(m_nn, w_test, W_grid_nn[l], m_nn_grid, W_grid_nn, Pot_nn_grid))
                plt.plot(W_grid_nn,Func_,c=color[k],label=str(round(m_nn,5)))

                
                Func_=[]
                for l in range(N_W):
                    Func_=np.append(Func_,interp3d_numba(m_nn, w_test, W_grid_nn[l], m_nn_grid, W_grid_nn, Pot_Markov_grid))
                plt.plot(W_grid_nn,Func_,linestyle='--',c=color[k])
            plt.legend()

            plt.show()
    
    
            dm=0.01
            for k in range(N_m_test):
                m_nn=m_nn_minus+(m_nn_plus-m_nn_minus)*(k/N_m_test)
                
                
                Func_=[]
                for l in range(N_W):
                    a=+interp3d_numba(m_nn+dm, w_test, W_grid_nn[l], m_nn_grid, W_grid_nn, Pot_nn_grid)/dm\
                      -interp3d_numba(m_nn+0 , w_test, W_grid_nn[l], m_nn_grid, W_grid_nn, Pot_nn_grid)/dm  
                    Func_=np.append(Func_,a)
                plt.plot(W_grid_nn,Func_,c=color[k],label=str(round(m_nn,5)))

                
                Func_=[]
                for l in range(N_W):
                    Func_=np.append(Func_,interp3d_numba(m_nn, w_test, W_grid_nn[l], m_nn_grid, W_grid_nn, Pot_Markov_grid))
                plt.plot(W_grid_nn,Func_,linestyle='--',c=color[k])
            plt.legend()
            plt.ylim([-15,10])
            plt.show()

    return Pot_nn_grid,W_grid_nn,m_nn_grid

######################################
######################################





##### Energies #####
####################

#######  X  #########
@numba.jit(nopython=True, fastmath=True)   
def Field_x_initialization(h_mat, X,N):
    Field_X_new=np.zeros(N)
    
    #### Fields for i=1,...,N-2
    for i in range(1,N-1):
        for j in range(0,i):
            Field_X_new[i]+=X[j]*h_mat[j,i]
            
        for j in range(i+1,N):
            Field_X_new[i]+=X[j]*h_mat[j,i]
            
    
    #### Field for i=0 
    for j in range(1,N):
        Field_X_new[0]+=X[j]*h_mat[j,0]
        
    #### Field for i=N-1
    for j in range(0,N-1):
        Field_X_new[N-1]+=X[j]*h_mat[j,N-1]
            
    return Field_X_new
    
@numba.jit(nopython=True, fastmath=True)   
def Field_x_update(h_mat, Field_X,X_old,X_new,index,N):

    #### Fields for i=1,...,N-2
    if index!=0 and index!=N-1:
        for i in range(index):
            Field_X[i]+=(X_new-X_old)*h_mat[i,index]
        
        for i in range(index+1,N):
            Field_X[i]+=(X_new-X_old)*h_mat[i,index]
            
    #### Field for i=0          
    if index==0:
        for i in range(1,N):
            Field_X[i]+=(X_new-X_old)*h_mat[i,0]
        
    #### Field for i=N-1    
    if index==N-1:
        for i in range(0,N-1):
            Field_X[i]+=(X_new-X_old)*h_mat[i,N-1]

@numba.jit(nopython=True, fastmath=True)               
def Flip_func(Field_X,X,index,N):
    b=np.random.rand()
    X_old=X[index]
    
    if np.exp(-2*Field_X[index]*X[index])>=b:
        X[index] *= -1.0
            
    X_new=X[index]
    return X_new,X_old



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
                


##### Overlaps #####
####################
@numba.jit(nopython=True, fastmath=True)               
def Overlaps_func(X,N):
    Overlaps_=np.zeros((N,N))
    for i in range(N):
            for j in range(N):
                Overlaps_[i,j]=X[i]*X[j]
    return Overlaps_

@numba.jit(nopython=True, fastmath=True)               
def Overlap_update_func(overlaps_avg,X,N,t_avg):
    
    overlaps_new    = Overlaps_func(X,N)
    
    for i in range(N):
            for j in range(N):
                overlaps_avg[i,j]=(overlaps_avg[i,j]*(t_avg-1)+overlaps_new[i,j])/t_avg
                
    return overlaps_avg

@numba.jit(nopython=True, fastmath=True)               
def Overlap_nn_update_func(overlaps_nn,X,N,t_nn):
    overlaps_new    = Overlaps_func(X,N)
    
    for i in range(N-1):
        overlaps_nn[i]=(overlaps_nn[i]*(t_nn-1)+overlaps_new[i,i+1])/t_nn
                
    return overlaps_nn

@numba.jit(nopython=True, fastmath=True)    
def dPot_update_func(dPot_avg, Pot_nn_grid,m_nn_grid,W_grid,   W,N,C_mat,t_avg):
    
    dm=0.01
    for i in range(N-1):
        dPot_new=+interp3d_numba(C_mat[i,i+1]+dm, W[i], W[i+1], m_nn_grid,W_grid,Pot_nn_grid)/dm\
                 -interp3d_numba(C_mat[i,i+1]+0 , W[i], W[i+1], m_nn_grid,W_grid,Pot_nn_grid)/dm
                 
        dPot_avg[i]=(dPot_avg[i]*(t_avg-1)+dPot_new)/t_avg
                    
    return dPot_avg
        
####################
####################




##### Updates  #####
####################
def Nearest_neighbor_field_update(h_mat,h_nn,overlaps_nn,mo,N,No):

    guess=[np.arctanh(mo),0.1]
    
    def Fields(index,guess):
        def sol_field(x):
            h_nn_ =x[0]
            h_o_=x[1]
        
            eqo  = mo                - np.tanh(h_nn_  +np.arctanh(np.tanh(h_o_)*(np.tanh(h_nn_)**(No-1))))
            eq_nn= overlaps_nn[index] - np.tanh(h_o_   +np.arctanh((np.tanh(h_nn_)**No)))
            
            return [eqo,eq_nn]
        
        sol=optimize.fsolve(sol_field,guess)
        return sol
        
    h_nn_new=np.zeros(N)
    for i in range(N-1):
        a=Fields(i,guess)
        h_nn_new[i]=np.arctanh(np.tanh(a[0])**No)
        
    return h_nn_new
             
def Update_h_C(α,κ,N,mo,No,  Overlap_X,Overlap_W,dPot, C_mat,h_mat,h_nn):
    η=0.65
    
    C_mat_new=np.zeros((N,N))
    h_mat_new=np.zeros((N,N))
    
    C_inv=np.linalg.inv(C_mat)
    
    
    ##### Update of C #####
    @numba.jit(nopython=True, fastmath=True) 
    def Update_C(C_mat_new):
        for i in range(N):
            for j in range(N):
                C_mat_new[i,j]=Overlap_X[i,j]
            
    Update_C(C_mat_new)   

    #######################
    #######################
        
    
    ##### Update of H #####
    @numba.jit(nopython=True, fastmath=True) 
    def Update_h(h_mat_new):
        for i in range(N):
            index=i
            for j in range(0,index):
                for k in range(N):
                    for l in range(N):
                        h_mat_new[i,j]+=+(α/2)*C_inv[l,i]*C_inv[j,k]*(Overlap_W[l,k]-C_mat[l,k])
                    
                    
            for j in range(index+1,N):
                for k in range(N):
                    for l in range(N):
                        h_mat_new[i,j]+=+(α/2)*C_inv[l,i]*C_inv[j,k]*(Overlap_W[l,k]-C_mat[l,k])
                 
        for i in range(N-1):  
            h_mat_new[i,i+1]+=-α*dPot[i]
            h_mat_new[i+1,i]+=-α*dPot[i]
    
    Update_h(h_mat_new) 
    
    #######################
    #######################
    
    
    ### Update of H_nn ####
    overlaps_nn=np.zeros(N)
    for i in range(N-1):
        overlaps_nn[i]=Overlap_X[i,i+1]
        

    h_nn_new=Nearest_neighbor_field_update(h_mat,h_nn,overlaps_nn,mo,N,No)        
    
    #######################
    #######################
                    
    return C_mat_new*η+(1-η)*C_mat,h_mat_new*η+(1-η)*h_mat,h_nn_new*η+(1-η)*h_nn

####################
####################


def Fields_dynamics(α,κ,N,T,mo,No, C_mat_0,h_mat_0,h_nn_0,read): 
    No=int(T/(N*(1-mo)))
    
    ### Initialization of fields and X ###
    X0    = (2*np.random.randint(2, size=N)-1) #np.ones(N)#
    h_mat = h_mat_0
    h_nn  = h_nn_0
    
    ### Initialization of fields ###
    W0    = np.zeros(N)
    C_mat = C_mat_0
        

    N_W=400
    N_m_nn=50
    
    if read==0:
        Pot_nn_grid,W_grid,m_nn_grid=Construct_all_nn_interaction(κ_plus=κ,κ_minus=-κ,\
                                                                  m_nn_minus=mo**No,m_nn_plus=mo,\
                                                                  N_m_nn=N_m_nn,N_W=N_W,mo=mo,No=No)
        Pot_nn_grid_sym=np.zeros((N_m_nn,N_W,N_W))
        file=open('Pot_nn_kappa_'+str(κ)+'_'+str(T)+'_'+str(N)+'.txt','w')  
        file.write('mo='+str(mo)+'    No='+str(No))
        file.write("\n")

        for i in range(N_m_nn):
            for j in range(N_W):
                for k in range(N_W):
                    
                    file.write(str(m_nn_grid[i]))
                    file.write('		')
                    file.write(str(W_grid[j]))
                    file.write('		')
                    file.write(str(W_grid[k]))
                    file.write('		')
                    file.write(str(Pot_nn_grid[i,j,k]))
                    file.write("\n")
                    
                    Pot_nn_grid_sym[i,j,k]=+(Pot_nn_grid[i,j,k]            +Pot_nn_grid[i,k,j]            )/4  \
                                           +(Pot_nn_grid[i,N_W-1-j,N_W-1-k]+Pot_nn_grid[i,N_W-1-k,N_W-1-j])/4
                    
        file.close()  

    if read==1:    
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
                    
    if read==-1:
        W_grid=np.linspace(-κ,κ,N_W)
        m_nn_grid=np.linspace(mo**No,0.9999,N_m_nn)
        Pot_nn_grid_sym=np.zeros((N_m_nn,N_W,N_W)) 
        
    N_iter_max=35
    for k in range(N_iter_max):
        print('κ:',κ,'  α:',α)
        print('iteration', k+1,' // ' ,N_iter_max )
        
        
        
        ##### MC of the binary contribution #####
        
        tolerance=1/350     ## Tolerance for the convergence of <X_i*X_j>
        sym_tolerance=1/35  ## Tolerance for  <X_i*X_j> being symmetric when doing a mirror symmetry in the middle of the chain
        t_update=1500*N
        t_max=900*t_update
        
        Overlap_X,X0      = Direct_Dynamics_ent(N,No,α,κ,mo,  C_mat,h_mat,h_nn  ,X0,tolerance,sym_tolerance,t_max)
        
        ########################################
        ########################################


        ##### MC of the margin contribution #####
        
        tolerance=1/550    ## Tolerance for the convergence of <W_i*W_j>
        t_update=1500*N
        t_max=900*t_update
        No_stop=0
        
        Overlap_W,dPot,W0 = Direct_Dynamics_ene(N,No,α,κ,mo,  C_mat,h_mat,h_nn  ,Pot_nn_grid_sym,m_nn_grid,W_grid,   W0,tolerance,t_max,No_stop)
        
        ########################################
        ########################################
   
   
        ################ Update ################
        
        C_mat,h_mat,h_nn=Update_h_C(α,κ,N,mo,No,  Overlap_X,Overlap_W,dPot, C_mat,h_mat,h_nn)
        
        ########################################
        ########################################
        
        if C_mat[0,N-1]>0.1:
            break
        
    return C_mat,h_mat,h_nn
    
def Direct_Dynamics_ent(N,No,α,κ,mo,  C_mat,h_mat,h_nn  ,X0,tolerance,sym_tolerance,t_max):
    
    h_mat_eff=np.copy(h_mat)
    for i in range(N-1):
        h_mat_eff[i,i+1]+=h_nn[i]
        h_mat_eff[i+1,i]+=h_nn[i]
        
    ### Initialization of spins ###
    X                = X0
    Field_X          = Field_x_initialization(h_mat_eff,X,N)

    overlaps_avg     = np.ones((N,N)) 
    overlaps_avg_mem = np.zeros((N,N)) 
    
    
    ### Param. for the MC ###
    Stop=0
    t=0
    t_print =3500*N
    t_mem_update=500*N
    
    while Stop==0:
        
        t+=1
        
        
        index = np.random.randint(N)                                # Select a random set of spins to flip
        X_new,X_old=Flip_func(Field_X,X,index,N)                    # Flip them according with prob exp(-h_i*X_i)
        if X_new!=X_old:
            Field_x_update(h_mat_eff, Field_X,X_old,X_new,index,N)  # Update the fields

        ### Update the memory ###
        if t%t_mem_update==0:
            
            if t>t_max:
                Stop=1
                
            if np.max(abs(overlaps_avg-overlaps_avg_mem))<tolerance:
               
                index=int(N/2)
                overlaps_middle    =np.roll(overlaps_avg[index,:],-index) 
                overlaps_middle_sym=np.roll((overlaps_middle[::-1]),1)
                
                if np.max(abs(overlaps_middle-overlaps_middle_sym))<sym_tolerance:
                    Stop=1
                
            overlaps_avg_mem=np.copy(overlaps_avg)
            
            
        overlaps_avg=Overlap_update_func(overlaps_avg,X,N,t)
        

        ### Plots ###
        #if t%t_print==0:
        if Stop==1:
            N_tot=N*No
            time_list =np.linspace(0,N_tot*(1-mo)/2,N_tot)
            time_list2=np.linspace(0,N_tot*(1-mo)/2,N)
            C_no_mem=mo**(np.linspace(0,N_tot,N_tot))
            
            plt.plot(time_list2, C_mat[0,:]            ,c='black',linestyle='--')
            plt.plot(time_list,  C_no_mem              ,c='grey', linestyle='-.')
            plt.plot(time_list2, overlaps_avg[0,:]                              )
            plt.plot(time_list2, overlaps_avg[N-1,::-1]                         )

        
            index=int(N_tot/2)
            index_=int(N/2)
            
            C_no_mem=mo**(abs(np.linspace(0,N_tot,N_tot)-index))
            C_middle            = C_mat[index_,:]
            overlaps_middle     = overlaps_avg[index_,:]
            overlaps_middle_sym = ((overlaps_middle[::-1]))

            
            plt.plot(time_list2,   C_middle,c='black',linestyle='--',label=r'$C(i,j)$')
            plt.plot(time_list,   C_no_mem,c='grey' ,linestyle='-.',label=r'$C_{\rm no-mem.}(i,j)$')
            plt.plot(time_list2,  overlaps_middle                  )
            plt.plot(time_list2,  overlaps_middle_sym              )
            plt.title('entropy contr.  (m='+str(round(mo,4))+'  kappa='+str(κ)+'  alpha='+str(α)+')')
            plt.ylabel(r'$\langle X_i X_j\rangle$')
            plt.xlabel(r'$\vert i-j \vert (1-m) $')
            plt.legend()
            plt.show()
            
            #####

    return  overlaps_avg,X

def Direct_Dynamics_ene(N,No,α,κ,mo,  C_mat,h_mat,h_nn  ,Pot_nn_grid,m_nn_grid,W_grid,   W0,tolerance,t_max,No_stop):   
    
    C_inv_mat=linalg.inv(C_mat)
    
    ### Initialization of spins ###
    W                = W0
    Field_W          = Field_w_initialization(C_inv_mat,W,N)
    
    overlaps_avg     = np.ones((N,N)) 
    overlaps_avg_mem = np.zeros((N,N)) 
    dPot_avg         = np.ones(N) 
    

    ### Param. for the MC ###
    Stop=0
    t=0
    t_print =50000*N
    t_mem_update=4500*N
    
    
    while Stop==0:
        t+=1
        
        i = np.random.randint(N)                                                              # Select a random margin to change
        Biased_W_sampling(Pot_nn_grid,m_nn_grid,W_grid,   κ,-κ,W,C_inv_mat,C_mat,Field_W,i)   # Update themargins and their field
            
        
        
        
        ### Update the memory ###
        if t%t_mem_update==0 and No_stop==0:
            if t>t_max:
                Stop=1
                
            if np.max(abs(overlaps_avg-overlaps_avg_mem))<tolerance:
                Stop=1
                
            overlaps_avg_mem=np.copy(overlaps_avg)
            
        overlaps_avg=Overlap_update_func(overlaps_avg,W,N,t)
        dPot_avg    =dPot_update_func(dPot_avg, Pot_nn_grid,m_nn_grid,W_grid,   W,N,C_mat,t)
        
    
            
        #if t%t_print==0:
        if Stop==1:

            N_tot=N*No
            
            time_list=np.linspace(0,N_tot*(1-mo)/2,N)
            #plt.plot(time_list , C_mat[0,:]            ,c='grey',linestyle='-.')
            plt.plot(time_list , overlaps_avg[0,:]                      )
            plt.plot(time_list , overlaps_avg[N-1,::-1]  ,linestyle='-.')            
        
            index=int(N/2)
            C_middle            = C_mat[index,:]
            overlaps_middle     = overlaps_avg[index,:]
            overlaps_middle_sym = np.roll((overlaps_middle[::-1]),1)
            
            #plt.plot(time_list , C_middle,c='grey',linestyle='-.',label='C(i,j)')
            plt.plot(time_list , overlaps_middle                  )
            plt.plot(time_list , overlaps_middle_sym              )         
            plt.title('energy contr.  (m='+str(round(mo,4))+'  kappa='+str(κ)+'  alpha='+str(α)+')')
            plt.ylabel(r'$\langle W_iW_j\rangle$')
            plt.xlabel(r'$\vert i-j \vert (1-m) $')
            #plt.legend()
            plt.show()
            
            
            N_tot=N*No
            time_list=np.linspace(0,N_tot*(1-mo),N-1)
            plt.plot(time_list , dPot_avg[0:N-1]    )
            plt.ylabel(r'$d\phi/dC$')
            plt.xlabel(r'$\vert i-j \vert (1-m) $')
            plt.show()
            
    return  overlaps_avg,dPot_avg,W






##### Time grid #####
T_tot=30
N=200
mo=0.9995
No=int(T_tot/(N*(1-mo)))

print('!! mo**No:',mo**No,'  mo:',mo,'  No:',No)
print('')
print('')
print('')
print('')

#####################
#####################


##### Initialization of C and h #####
C_mat_0=np.zeros((N,N))
h_mat_0=np.zeros((N,N))
h_nn_0   =np.zeros(N)

for i in range(N):
    for j in range(N):
        C_mat_0[i,j]=(mo**No)**(abs(i-j))
        h_mat_0[i,j]=0.0001**(abs(i-j))
            
    h_mat_0[i,i]=0
            
for i in range(N-1):
    h_nn_0[i]=np.arctanh(mo**No)
    
#####################
#####################








κ  = 0.35
α  = 0.2
dα = 0.025

    
for k in range(6000):
    α+=dα
    if k==0:
        read=1
    else:
        read=1
    C_mat_0,h_mat_0,h_nn_0=Fields_dynamics(α,κ,N,T_tot,mo,No, C_mat_0,h_mat_0,h_nn_0,read)
    
    
    file_1=open('Correlation_high_res_zoom_zoom(kappa='+str(κ)+'_alpha='+str(α)+').txt','w')    
    file_1.write('T='+str(T_tot)+' N='+str(N)+' mo='+str(mo)+' No='+str(No))
    file_1.write("\n")
    for i in range(N):
        for j in range(N):
            file_1.write(str(C_mat_0[i,j]))
            file_1.write('     ')
        file_1.write("\n")
    file_1.close()

    file_2=open('Fields_high_res_zoom_zoom(kappa='+str(κ)+'_alpha='+str(α)+').txt','w')    
    file_2.write('T='+str(T_tot)+' N='+str(N)+' mo='+str(mo)+' No='+str(No))
    file_2.write("\n")
    for i in range(N):
        for j in range(N):
            file_2.write(str(h_mat_0[i,j]))
            file_2.write('     ')
        file_2.write("\n")
    file_2.close()
    
    file_3=open('Fields_nn_high_res_zoom_zoom(kappa='+str(κ)+'_alpha='+str(α)+').txt','w')    
    file_3.write('T='+str(T_tot)+' N='+str(N)+' mo='+str(mo)+' No='+str(No))
    file_3.write("\n")
    for i in range(N):
        file_3.write(str(h_nn_0[i]))
        file_3.write('     ')
    file_3.close()
    
    if C_mat_0[0,N-1]>0.1:
        break
    






    
    










