
import numpy as np
import torch
import networkx as nx
import numpy as np
import networkx as nx
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.CBN import CausalBayesianNetwork as CBN
import operations as ops

def monge_map(m_alpha, Sigma_alpha, m_beta, Sigma_beta):
    """
    Compute the Monge map between two multivariate Gaussians and return the transformation
    function T(x) along with the matrix T.

    Args:
    - m_alpha: Mean of the first Gaussian (m_alpha has shape (l,))
    - Sigma_alpha: Covariance matrix of the first Gaussian (Sigma_alpha has shape (l, l))
    - m_beta: Mean of the second Gaussian (m_beta has shape (l,))
    - Sigma_beta: Covariance matrix of the second Gaussian (Sigma_beta has shape (l, l))

    Returns:
    - T(x): A function that applies the Monge map transformation to a point x from the first Gaussian
    - T: The matrix T used for the transformation (T has shape (l, l))
    """
    # Step 1: Compute A using the formula
    Sigma_alpha_half = np.linalg.cholesky(Sigma_alpha)  # Cholesky decomposition of Sigma_alpha
    Sigma_alpha_half_inv = np.linalg.inv(Sigma_alpha_half)  # Inverse of Sigma_alpha_half

    # Compute A using the given formula
    A = Sigma_alpha_half_inv @ np.linalg.cholesky(Sigma_alpha_half.T @ Sigma_beta @ Sigma_alpha_half) @ Sigma_alpha_half_inv.T

    # Step 2: Define the Monge map as a function T(x) = m_beta + A(x - m_alpha)
    def tau(x):
        return m_beta + A @ (x - m_alpha)

    return tau, A


# Proximal operator of a matrix frobenious norm
def prox_operator(A, lambda_param):
    frobenius_norm = torch.norm(A, p='fro')
    scaling_factor = torch.max(1 - lambda_param / frobenius_norm, torch.zeros_like(frobenius_norm))
    return scaling_factor * A

def diagonalize(A):
    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eig(A)  
    eigvals_real     = eigvals.real  
    eigvals_real     = torch.sqrt(eigvals_real)  # Take the square root of the eigenvalues

    return torch.diag(eigvals_real)

# def sqrtm_svd(A):
#     # Compute the SVD of A
#     U, S, V = torch.svd(A)
    
#     # Take the square root of the singular values
#     S_sqrt = torch.sqrt(torch.clamp(S, min=0.0))  # Ensure non-negative singular values
    
#     # Reconstruct the square root matrix
#     sqrt_A = U @ torch.diag(S_sqrt) @ V.T
    
#     return sqrt_A
def sqrtm_svd_np(A):
    """
    Compute the matrix square root using SVD for numpy arrays.
    
    Args:
        A: numpy array of shape (n x n)
        
    Returns:
        Matrix square root of A
    """
    # Handle non-finite values
    A = np.nan_to_num(A, nan=1e-6, posinf=1e10, neginf=-1e10)
    
    # Ensure matrix is symmetric
    A = 0.5 * (A + A.T)
    
    # Add small regularization term
    eps = 1e-10
    A = A + eps * np.eye(A.shape[0])
    
    try:
        # Try SVD computation
        U, S, V = np.linalg.svd(A)
        
        # Ensure numerical stability of singular values
        S = np.clip(S, eps, None)
        S_sqrt = np.sqrt(S)
        
        return U @ np.diag(S_sqrt) @ V
        
    except Exception as e:
        print(f"SVD failed: {e}")
        # Return regularized identity matrix as fallback
        return np.eye(A.shape[0]) * np.linalg.norm(A)
    
def sqrtm_svd(A):
    # Handle non-finite values
    A = torch.nan_to_num(A, nan=1e-6, posinf=1e10, neginf=-1e10)
    
    # Ensure matrix is symmetric
    A = 0.5 * (A + A.T)
    
    # Add small regularization term
    eps = 1e-10
    A = A + eps * torch.eye(A.shape[0], device=A.device)
    
    try:
        # Try SVD computation
        U, S, V = torch.svd(A)
        
        # Ensure numerical stability of singular values
        S = torch.clamp(S, min=eps)
        S_sqrt = torch.sqrt(S)
        
        return U @ torch.diag(S_sqrt) @ V.T
        
    except Exception as e:
        print(f"SVD failed: {e}")
        # Return regularized identity matrix as fallback
        return torch.eye(A.shape[0], device=A.device) * torch.norm(A)


def are_matrices_equal(matrix1, matrix2, tol=1e-11):
    """
    Check if two matrices are equal within a given tolerance.
    
    Args:
        matrix1 (np.ndarray): The first matrix to compare.
        matrix2 (np.ndarray): The second matrix to compare.
        tol (float): Tolerance for element-wise comparison (default: 1e-8).
    
    Returns:
        bool: True if the matrices are equal within the given tolerance, False otherwise.
    """
    if matrix1.shape != matrix2.shape:
        return False  # Matrices must have the same shape to be equal

    return torch.allclose(matrix1, matrix2, atol=tol)

def regmat(matrix, eps=1e-10):
    # Replace NaN and Inf values with finite numbers
    matrix_new = torch.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)
    # if not are_matrices_equal(matrix, matrix_new):
    #     print('O')
    # Add a small epsilon to the diagonal for numerical stability
    if matrix_new.dim() == 2 and matrix_new.size(0) == matrix_new.size(1):
        matrix_new = matrix_new + eps * torch.eye(matrix_new.size(0), device=matrix_new.device)
    
    return matrix_new

def sqrtm_eig(A):
    eigvals, eigvecs = torch.linalg.eig(A)
    eigvals_real = eigvals.real
    
    # Ensure eigenvalues are non-negative for the square root to be valid
    eigvals_sqrt = torch.sqrt(torch.clamp(eigvals_real, min=0.0))  # Square root of non-negative eigenvalues

    # Reconstruct the square root of the matrix using the eigenvectors
    # Make sure the eigenvectors are also real
    eigvecs_real = eigvecs.real
    
    # Reconstruct the matrix square root
    sqrt_A = eigvecs_real @ torch.diag(eigvals_sqrt) @ eigvecs_real.T
    
    return sqrt_A


def plot_epoch_progress(epoch, epoch_min_objectives, epoch_max_objectives, robust):
    """Plot optimization progress for current epoch"""
    if robust:
        # Create subplot with both min and max
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=(f'Minimization Steps (Epoch {epoch+1})', 
                                         f'Maximization Steps (Epoch {epoch+1})'))
        
        # Add min step objectives
        fig.add_trace(
            go.Scatter(y=epoch_min_objectives, 
                      mode='lines', 
                      name='Min steps',
                      line=dict(color='green')),
            row=1, col=1
        )

        # Add max step objectives
        fig.add_trace(
            go.Scatter(y=epoch_max_objectives, 
                      mode='lines', 
                      name='Max steps',
                      line=dict(color='purple')),
            row=1, col=2
        )
        
        # Update both axes
        for row, col in [(1,1), (1,2)]:
            fig.update_xaxes(
                title_text="Step", 
                row=row, col=col,
                showgrid=True,
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='black'
            )
        
        fig.update_yaxes(
            title_text="Objective T Value", 
            row=1, col=1,
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        fig.update_yaxes(
            title_text="Objective θ Value", 
            row=1, col=2,
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        
    else:
        # Create single plot for minimization only
        fig = go.Figure()
        
        # Add min step objectives
        fig.add_trace(
            go.Scatter(y=epoch_min_objectives, 
                      mode='lines', 
                      name='Min steps',
                      line=dict(color='green'))
        )
        
        # Update axes for single plot
        fig.update_xaxes(
            title_text="Step",
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        fig.update_yaxes(
            title_text="Objective T Value",
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )

    # Common layout updates
    fig.update_layout(
        height=400,
        width=1000 if robust else 500,
        showlegend=True,
        title_text=f"Optimization Progress - Epoch {epoch+1}",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )

    fig.show()

def plot_overall_progress(epoch_objectives, robust):
    """Plot overall optimization progress"""
    if robust:
        # Create subplot with both objectives
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('T Objective across Epochs', 
                                         'θ Objective across Epochs'))
        
        # Plot T objectives
        fig.add_trace(
            go.Scatter(y=epoch_objectives['T_objectives'], 
                      mode='lines', 
                      name='T objective',
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # Plot theta objectives
        fig.add_trace(
            go.Scatter(y=epoch_objectives['theta_objectives'], 
                      mode='lines', 
                      name='θ objective',
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Update both sets of axes
        for row in [1, 2]:
            fig.update_xaxes(
                title_text="Epoch", 
                row=row, col=1,
                showgrid=True,
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='black'
            )
            
            title = "T Objective Value" if row == 1 else "θ Objective Value"
            fig.update_yaxes(
                title_text=title, 
                row=row, col=1,
                showgrid=True,
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='black'
            )
            
    else:
        # Create single plot for T objective only
        fig = go.Figure()
        
        # Plot T objectives
        fig.add_trace(
            go.Scatter(y=epoch_objectives['T_objectives'], 
                      mode='lines', 
                      name='T objective',
                      line=dict(color='green'))
        )
        
        # Update axes for single plot
        fig.update_xaxes(
            title_text="Epoch",
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        fig.update_yaxes(
            title_text="T Objective Value",
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )

    # Common layout updates
    fig.update_layout(
        height=800 if robust else 400,
        width=1000 if robust else 500,
        showlegend=True,
        title_text="Overall Optimization Progress",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )

    fig.show()

def print_results(T, paramsL, paramsH, elapsed_time):
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS".center(50))
    print("="*50 + "\n")

    print("Final Transformation Matrix T:")
    print("-"*30)
    print(np.array2string(T, precision=4, suppress_small=True))
    print(f"\nCondition Number: {condition_number(T):.4f}")
    
    print("\nLow-Level Parameters:")
    print("-"*30)
    print(f"μ_L = {np.array2string(paramsL['mu_U'], precision=4, suppress_small=True)}")
    print(f"Σ_L = \n{np.array2string(paramsL['Sigma_U'], precision=4, suppress_small=True)}")
    
    print("\nHigh-Level Parameters:")
    print("-"*30)
    print(f"μ_H = {np.array2string(paramsH['mu_U'], precision=4, suppress_small=True)}")
    print(f"Σ_H = \n{np.array2string(paramsH['Sigma_U'], precision=4, suppress_small=True)}")

    print(f"\nOptimization time: {elapsed_time:.4f} seconds")

def check_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta):
    """
    Check if the given mu_L, Sigma_L, mu_H, Sigma_H satisfy the constraints and return the violation amount if any.

    Arguments:
    mu_L, Sigma_L, mu_H, Sigma_H: Input tensors for the model parameters.
    hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H: Target tensors.
    epsilon, delta: Constraint thresholds.

    Returns:
    bool: True if both constraints are satisfied, False otherwise.
    violation_1, violation_2: Violation amounts for each constraint.
    """
    # 1st constraint: epsilon^2 - ||mu_L - hat_mu_L||^2_2 - ||Sigma_L^{1/2} - hat_Sigma_L^{1/2}||^2_2 >= 0
    mu_L_term = torch.norm(mu_L - hat_mu_L) ** 2
    Sigma_L_sqrt = sqrtm_svd(Sigma_L)  # Assuming Sigma_L is positive-definite
    hat_Sigma_L_sqrt = sqrtm_svd(hat_Sigma_L)  # Assuming hat_Sigma_L is positive-definite
    Sigma_L_term = torch.norm(Sigma_L_sqrt - hat_Sigma_L_sqrt) ** 2

    # First constraint check and violation amount
    constraint_1 = epsilon ** 2 - mu_L_term - Sigma_L_term
    violation_1 = max(0, -constraint_1.item())  # If violated, return how much negative it is (violation)

    # 2nd constraint: delta^2 - ||mu_H - hat_mu_H||^2_2 - ||Sigma_H^{1/2} - hat_Sigma_H^{1/2}||^2_2 >= 0
    mu_H_term = torch.norm(mu_H - hat_mu_H) ** 2
    Sigma_H_sqrt = sqrtm_svd(Sigma_H)  # Assuming Sigma_H is positive-definite
    hat_Sigma_H_sqrt = sqrtm_svd(hat_Sigma_H)  # Assuming hat_Sigma_H is positive-definite
    Sigma_H_term = torch.norm(Sigma_H_sqrt - hat_Sigma_H_sqrt) ** 2

    # Second constraint check and violation amount
    constraint_2 = delta ** 2 - mu_H_term - Sigma_H_term
    violation_2 = max(0, -constraint_2.item())  # If violated, return how much negative it is (violation)

    # Return True if both constraints are satisfied, False otherwise, along with violations
    if violation_1 == 0 and violation_2 == 0:
        return True, violation_1, violation_2
    else:
        return False, violation_1, violation_2

def check_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta):
    # Constraint 1: epsilon^2 - ||mu_L - hat_mu_L||_2^2 - ||Sigma_L^{1/2} - hat_Sigma_L^{1/2}||_2^2 >= 0
    constraint_L = epsilon**2 - (torch.norm(mu_L - hat_mu_L)**2) - (torch.norm(sqrtm_svd(Sigma_L) - sqrtm_svd(hat_Sigma_L))**2)
    
    # Constraint 2: delta^2 - ||mu_H - hat_mu_H||_2^2 - ||Sigma_H^{1/2} - hat_Sigma_H^{1/2}||_2^2 >= 0
    constraint_H = delta**2 - (torch.norm(mu_H - hat_mu_H)**2) - (torch.norm(sqrtm_svd(Sigma_H) - sqrtm_svd(hat_Sigma_H))**2)
    
    # Return whether constraints are satisfied (i.e., >= 0) and the constraint violations
    return constraint_L, constraint_H


def enforce_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta):
    constraint_L, constraint_H = check_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta)
    
    # Clip values if constraints are violated
    if constraint_L < 0:
        #print(f"Constraint for mu_L and Sigma_L violated. Fixing...")
        mu_L = hat_mu_L + torch.clamp(mu_L - hat_mu_L, min=-epsilon, max=epsilon)
        Sigma_L = hat_Sigma_L + torch.clamp(Sigma_L - hat_Sigma_L, min=-epsilon, max=epsilon)
    
    if constraint_H < 0:
        #print(f"Constraint for mu_H and Sigma_H violated. Fixing...")
        mu_H = hat_mu_H + torch.clamp(mu_H - hat_mu_H, min=-delta, max=delta)
        Sigma_H = hat_Sigma_H + torch.clamp(Sigma_H - hat_Sigma_H, min=-delta, max=delta)
    
    return mu_L, Sigma_L, mu_H, Sigma_H


# # Updates and opt routines
# def update_mu_L(T, mu_L, mu_H, LLmodels, HLmodels, Ill, Ihl, omega, lambda_L, hat_mu_L, eta):
#     grad_mu_L = torch.zeros_like(mu_L, dtype=torch.float32) 
#     for n, iota in enumerate(Ill):
#         L_i = torch.from_numpy(LLmodels[iota].compute_mechanism()).float() 
#         V_i = T @ L_i  
#         H_i = torch.from_numpy(HLmodels[omega[iota]].compute_mechanism()).float() 

#         grad_mu_L += torch.matmul(V_i.T, torch.matmul(V_i, mu_L.float()) - torch.matmul(H_i, mu_H.float())) 
    
#     grad_mu_L = (2 / n) * grad_mu_L - 2 * lambda_L * (mu_L - hat_mu_L)
#     mu_L = mu_L + (eta * grad_mu_L)
#     return mu_L

# def update_mu_H(T, mu_L, mu_H, LLmodels, HLmodels, Ill, Ihl, omega, lambda_H, hat_mu_H, eta):
#     grad_mu_H = torch.zeros_like(mu_H, dtype=torch.float32)  
#     for n, iota in enumerate(Ill):
#         L_i = torch.from_numpy(LLmodels[iota].compute_mechanism()).float()  
#         V_i = T @ L_i  
#         H_i = torch.from_numpy(HLmodels[omega[iota]].compute_mechanism()).float()  

#         grad_mu_H -= torch.matmul(H_i.T, torch.matmul(V_i, mu_L.float()) - torch.matmul(H_i, mu_H.float()))
    
#     grad_mu_H = (2 / n) * grad_mu_H - 2 * lambda_H * (mu_H - hat_mu_H)
    
#     mu_H = mu_H + (eta * grad_mu_H)
#     return mu_H


# def update_Sigma_L_half(T, Sigma_L, LLmodels, Ill, Ihl, omega, lambda_L, hat_Sigma_L, eta):
#     grad_Sigma_L = torch.zeros_like(Sigma_L)
#     term1 = torch.zeros_like(Sigma_L)
#     for n, iota in enumerate(Ill):
#         L_i = torch.from_numpy(LLmodels[iota].compute_mechanism())
#         V_i = T @ L_i.float()
#         term1 = term1 + torch.matmul(V_i.T, V_i)

#     Sigma_L_sqrt     = sqrtm_svd(Sigma_L)  
#     hat_Sigma_L_sqrt = sqrtm_svd(hat_Sigma_L) 

#     term2 = -2 * lambda_L * (Sigma_L_sqrt - hat_Sigma_L_sqrt) @ torch.inverse(Sigma_L_sqrt)

#     grad_Sigma_L = (2 / n) * term1 + term2

#     Sigma_L_half = Sigma_L + eta * grad_Sigma_L
#     #Sigma_L_half  = diagonalize(Sigma_L_half)
#     return Sigma_L_half


# def update_Sigma_L(T, Sigma_L_half, LLmodels, Ill, Ihl, omega, Sigma_H, HLmodels, lambda_param):
#     Sigma_L_final = torch.zeros_like(Sigma_L_half, dtype=torch.float32)  
#     for n, iota in enumerate(Ill):
#         L_i = torch.from_numpy(LLmodels[iota].compute_mechanism()).float()  
#         V_i = T @ L_i  
#         H_i = torch.from_numpy(HLmodels[omega[iota]].compute_mechanism()).float()  
        
#         Sigma_L_half      = Sigma_L_half.float()
#         V_Sigma_V         = torch.matmul(V_i, torch.matmul(Sigma_L_half, V_i.T))
#         sqrtm_V_Sigma_V   = sqrtm_svd(regmat(V_Sigma_V))
#         prox_Sigma_L_half = torch.matmul(prox_operator(sqrtm_V_Sigma_V, lambda_param), prox_operator(sqrtm_V_Sigma_V, lambda_param).T)
 
#         ll_term_a = torch.matmul(regmat(torch.linalg.pinv(V_i)), regmat(prox_Sigma_L_half))
#         ll_term_b = torch.linalg.pinv(V_i).T
#         ll_term   = torch.matmul(ll_term_a, ll_term_b)
#         #ll_term           = torch.matmul(torch.matmul(torch.linalg.pinv(V_i), oput.regmat(prox_Sigma_L_half)), torch.linalg.pinv(V_i).T)

#         Sigma_H   = Sigma_H.float()  
#         H_Sigma_H = torch.matmul(H_i, torch.matmul(Sigma_H, H_i.T)).float()
#         hl_term   = torch.norm(sqrtm_svd(regmat(H_Sigma_H)), p='fro')

#         Sigma_L_final = Sigma_L_final + (ll_term * hl_term)

#     Sigma_L_final = Sigma_L_final * (2 / n)
#     Sigma_L_final = diagonalize(Sigma_L_final)

#     return Sigma_L_final


# def update_Sigma_H_half(T, Sigma_H, HLmodels, Ill, Ihl, omega, lambda_H, hat_Sigma_H, eta):
#     grad_Sigma_H = torch.zeros_like(Sigma_H)
#     term1        = torch.zeros_like(Sigma_H)
#     for n, kappa in enumerate(Ihl):
#         H_i   = torch.from_numpy(HLmodels[kappa].compute_mechanism()).float()
#         term1 = term1 + torch.matmul(H_i.T, H_i)

#     Sigma_H_sqrt     = sqrtm_svd(Sigma_H)  
#     hat_Sigma_H_sqrt = sqrtm_svd(hat_Sigma_H) 

#     term2 = -2 * lambda_H * (Sigma_H_sqrt - hat_Sigma_H_sqrt) @ torch.inverse(Sigma_H_sqrt)

#     grad_Sigma_H = (2 / n) * term1 + term2

#     Sigma_H_half = Sigma_H + eta * grad_Sigma_H
#     return Sigma_H_half

# def check_for_invalid_values(matrix):
#     if torch.isnan(matrix).any() or torch.isinf(matrix).any():
#         #print("Matrix contains NaN or Inf values!")
#         return True
#     return False

# def handle_nans(matrix, replacement_value=0.0):
#     # Replace NaNs with a given value (default is 0)
#     if torch.isnan(matrix).any():
#         print("Warning: NaN values found! Replacing with zero.")
#         matrix = torch.nan_to_num(matrix, nan=replacement_value)
#     return matrix


# def update_Sigma_H(T, Sigma_H_half, LLmodels, Ill, Ihl, omega, Sigma_L, HLmodels, lambda_param):
#     if check_for_invalid_values(Sigma_L):
#         print("Sigma_L contains NaN or Inf values!")
#     Sigma_H_final = torch.zeros_like(Sigma_H_half)
#     for n, iota in enumerate(Ill):
#         L_i = torch.from_numpy(LLmodels[iota].compute_mechanism())
#         V_i = T @ L_i.float()
#         H_i = torch.from_numpy(HLmodels[omega[iota]].compute_mechanism()).float()

#         H_Sigma_H         = torch.matmul(H_i, torch.matmul(Sigma_H_half, H_i.T))
#         sqrtm_H_Sigma_H   = sqrtm_svd(regmat(H_Sigma_H))
#         prox_Sigma_H_half = torch.matmul(prox_operator(sqrtm_H_Sigma_H, lambda_param), prox_operator(sqrtm_H_Sigma_H, lambda_param).T)
#         hl_term           = torch.matmul(torch.matmul(torch.inverse(H_i), regmat(prox_Sigma_H_half)), torch.inverse(H_i).T)  
        
#         V_Sigma_V = torch.matmul(V_i, torch.matmul(Sigma_L, V_i.T))
#         ll_term   = torch.norm(sqrtm_svd(regmat(V_Sigma_V)))

#         Sigma_H_final = Sigma_H_final + (ll_term * hl_term)
    
#     Sigma_H_final = Sigma_H_final * (2 / n)
#     Sigma_H_final = diagonalize(Sigma_H_final)
    
#     return Sigma_H_final

# def check_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta):
#     # Constraint 1: epsilon^2 - ||mu_L - hat_mu_L||_2^2 - ||Sigma_L^{1/2} - hat_Sigma_L^{1/2}||_2^2 >= 0
#     constraint_L = epsilon**2 - (torch.norm(mu_L - hat_mu_L)**2) - (torch.norm(sqrtm_svd(Sigma_L) - sqrtm_svd(hat_Sigma_L))**2)
    
#     # Constraint 2: delta^2 - ||mu_H - hat_mu_H||_2^2 - ||Sigma_H^{1/2} - hat_Sigma_H^{1/2}||_2^2 >= 0
#     constraint_H = delta**2 - (torch.norm(mu_H - hat_mu_H)**2) - (torch.norm(sqrtm_svd(Sigma_H) - sqrtm_svd(hat_Sigma_H))**2)
    
#     # Return whether constraints are satisfied (i.e., >= 0) and the constraint violations
#     return constraint_L, constraint_H


# def enforce_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta):
#     constraint_L, constraint_H = check_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta)
    
#     # Clip values if constraints are violated
#     if constraint_L < 0:
#         print(f"Constraint for mu_L and Sigma_L violated. Fixing...")
#         mu_L = hat_mu_L + torch.clamp(mu_L - hat_mu_L, min=-epsilon, max=epsilon)
#         Sigma_L = hat_Sigma_L + torch.clamp(Sigma_L - hat_Sigma_L, min=-epsilon, max=epsilon)
    
#     if constraint_H < 0:
#         print(f"Constraint for mu_H and Sigma_H violated. Fixing...")
#         mu_H = hat_mu_H + torch.clamp(mu_H - hat_mu_H, min=-delta, max=delta)
#         Sigma_H = hat_Sigma_H + torch.clamp(Sigma_H - hat_Sigma_H, min=-delta, max=delta)
    
#     return mu_L, Sigma_L, mu_H, Sigma_H

# def optimize_max(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, Ill, Ihl, omega, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, lambda_L, lambda_H, lambda_param, eta, num_steps_max, epsilon, delta, seed):
    
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     for t in range(num_steps_max): 
#         mu_L         = update_mu_L(T, mu_L, mu_H, LLmodels, HLmodels, Ill, Ihl, omega, lambda_L, hat_mu_L, eta)
#         mu_H         = update_mu_H(T, mu_L, mu_H, LLmodels, HLmodels, Ill, Ihl, omega, lambda_H, hat_mu_H, eta)
#         Sigma_L_half = update_Sigma_L_half(T, Sigma_L, LLmodels, Ill, Ihl, omega, lambda_L, hat_Sigma_L, eta)
#         Sigma_L      = update_Sigma_L(T, Sigma_L_half, LLmodels, Ill, Ihl, omega, Sigma_H, HLmodels, lambda_param)
#         Sigma_H_half = update_Sigma_H_half(T, Sigma_H, HLmodels, Ill, Ihl, omega, lambda_H, hat_Sigma_H, eta)
#         Sigma_H      = update_Sigma_H(T, Sigma_H_half, LLmodels, Ill, Ihl, omega, Sigma_L, HLmodels, lambda_param)
        
#         # Project onto Gelbrich balls
#         mu_L, Sigma_L = project_onto_gelbrich_ball(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
#         mu_H, Sigma_H = project_onto_gelbrich_ball(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)
        
#         # Verify constraints
#         satisfied_L, dist_L, epsi = verify_gelbrich_constraint(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
#         satisfied_H, dist_H, delt = verify_gelbrich_constraint(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)
        
#         if not satisfied_L:
#             print(f"Warning: Constraints not satisfied for mu_L and Sigma_L! Distance: {dist_L} and epsilon = {epsi}")

#         if not satisfied_H:
#             print(f"Warning: Constraints not satisfied for mu_H and Sigma_H! Distance: {dist_H} and delta = {delt}")

#         #mu_L, Sigma_L, mu_H, Sigma_H = enforce_constraints(mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, epsilon, delta)
        
#         obj = 0
        
#         for i, iota in enumerate(Ill):
#             L_i = torch.from_numpy(LLmodels[iota].compute_mechanism())
#             V_i = T @ L_i.float()
#             H_i = torch.from_numpy(HLmodels[omega[iota]].compute_mechanism()).float()
                        
#             L_i_mu_L = V_i @ mu_L
#             H_i_mu_H = H_i @ mu_H
#             term1 = torch.norm(L_i_mu_L.float() - H_i_mu_H.float())**2
            
#             V_Sigma_V = regmat(V_i.float() @ Sigma_L.float() @ V_i.T.float())
#             H_Sigma_H = regmat(H_i.float() @ Sigma_H.float() @ H_i.T.float())

#             term2 = torch.trace(V_Sigma_V)
#             term3 = torch.trace(H_Sigma_H)
            
#             sqrtVSV = sqrtm_svd(V_Sigma_V)
#             sqrtHSH = sqrtm_svd(H_Sigma_H)

#             term4 = -2 * torch.trace(sqrtm_svd(regmat(sqrtVSV @ sqrtHSH @ sqrtVSV)))
#             #term4 = -2 * torch.norm(sqrtVSV @ sqrtHSH, 'nuc')
            
#             obj = obj + (term1 + term2 + term3 + term4)
        
#         obj = obj/i
        
#         #print(f"Max step {t+1}/{num_steps_max}, Objective: {obj.item()}")

#     return obj, mu_L, Sigma_L, mu_H, Sigma_H

# def optimize_min(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels,  Ill, Ihl, omega, num_steps_min, optimizer_T, seed):

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     objective_T = 0 
#     for step in range(num_steps_min):
#         objective_T = 0  # Reset objective at the start of each step
#         for n, iota in enumerate(Ill):
#             L_i = torch.from_numpy(LLmodels[iota].compute_mechanism()).float()
#             H_i = torch.from_numpy(HLmodels[omega[iota]].compute_mechanism()).float()

#             L_i_mu_L = L_i @ mu_L  
#             H_i_mu_H = H_i @ mu_H 

#             term1 = torch.norm(T @ L_i_mu_L - H_i_mu_H) ** 2
#             term2 = torch.trace(T @ L_i @ Sigma_L @ L_i.T @ T.T)
#             term3 = torch.trace(H_i @ Sigma_H @ H_i.T)
            
#             L_i_Sigma_L = regmat(T @ L_i @ Sigma_L @ L_i.T @ T.T)
#             H_i_Sigma_H = regmat(H_i @ Sigma_H @ H_i.T)

#             #term4 = -2 * torch.norm(oput.sqrtm_svd(L_i_Sigma_L) @ oput.sqrtm_svd(H_i_Sigma_H), 'nuc')
#             term4 = -2 * torch.trace(sqrtm_svd(sqrtm_svd(L_i_Sigma_L) @ H_i_Sigma_H @ sqrtm_svd(L_i_Sigma_L)))

#             objective_T += term1 + term2 + term3 + term4

#         objective_T = objective_T/n

#         optimizer_T.zero_grad() 
#         objective_T.backward(retain_graph=True)  
#         # Log the gradient norm
#         grad_norm = T.grad.norm().item()
#         #print(f"Step {step+1}/{num_steps_min}: Objective = {objective_T.item()}, Gradient Norm = {grad_norm}")

#         # # Gradient clipping
#         # clip_grad_norm_([T], 1.0)

#         optimizer_T.step()      

#         #print(f"Min step {step+1}/{num_steps_min}, Objective: {objective_T.item()}")

#     return objective_T, T 

# def optimize_min_max(mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, 
#                      hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, Ill, Ihl, omega,
#                      epsilon, delta, lambda_L, lambda_H, lambda_param, 
#                      eta_min, eta_max, max_iter, num_steps_min, num_steps_max, tol, seed):
    
#     j = 0
#     torch.manual_seed(seed) 
#     torch.cuda.manual_seed_all(seed)

#     T           = torch.randn(mu_H.shape[0], mu_L.shape[0], requires_grad=True)
#     #optimizer_T = torch.optim.Adam([T], lr=0.001)
#     optimizer_T = torch.optim.Adam([T], lr=eta_min, eps=1e-8)

#     previous_objective       = float('inf')  
#     for epoch in tqdm(range(max_iter)):
#         #print("MINIMIZING T")
#         objective_T, T = optimize_min(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, Ill, Ihl, omega, num_steps_min, optimizer_T, seed)
        
#         #print()
#         #print("MAX mu_L, Sigma_L, mu_H, Sigma_H")
#         objective_theta, mu_L, Sigma_L, mu_H, Sigma_H = optimize_max(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels,  Ill, Ihl, omega, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
#                                                                          lambda_L, lambda_H, lambda_param, eta_max, num_steps_max, epsilon, delta, seed)
#         if contains_negative(Sigma_L) == True:
#             print('Sigma_L contains negative values')
#             print(Sigma_L)
#             print( )
#         if contains_negative(Sigma_H) == True:
#             print('Sigma_H contains negative values')
#             print(Sigma_H)
#             print( )

#         # Check for convergence by comparing the difference in objective values
#         criterion = abs(previous_objective - objective_T.item())
        
#         if criterion < tol:
#             print(f"Convergence reached at epoch {epoch+1} with objective {objective_T.item()}")
#             break

#         previous_objective = objective_T.item()

#     print("Final T:", T)
#     print("Final mu_L:", mu_L)
#     print("Final Sigma_L:", Sigma_L)
#     print("Final mu_H:", mu_H)
#     print("Final Sigma_H:", Sigma_H)

#     return mu_L, Sigma_L, mu_H, Sigma_H, T

def contains_negative(matrix):
    return (matrix < 0).any().item()

def project_onto_gelbrich_ball(mu, Sigma, hat_mu, hat_Sigma, epsilon, max_iter=100, tol=1e-6):
    """
    Project (mu, Sigma) onto the Gelbrich ball with detailed debugging
    """
    for i in range(max_iter):
        mu_dist_sq     = torch.sum((mu - hat_mu)**2)
        Sigma_sqrt     = sqrtm_svd(Sigma)
        hat_Sigma_sqrt = sqrtm_svd(hat_Sigma)
        Sigma_dist_sq  = torch.sum((Sigma_sqrt - hat_Sigma_sqrt)**2)
        
        G_squared      = mu_dist_sq + Sigma_dist_sq
        
        #print(f"Iteration {i}: G_squared = {G_squared.item()}, epsilon^2 = {epsilon**2}")
        
        if G_squared <= epsilon**2 + tol:
            #print(f"Projection converged in {i+1} iterations with G_squared = {G_squared.item()} and epsilon_sq = {epsilon**2}")
            break
        
        scale = epsilon / torch.sqrt(G_squared)

        mu         = hat_mu + scale * (mu - hat_mu)
        Sigma_diff = Sigma_sqrt - hat_Sigma_sqrt
        Sigma_sqrt = hat_Sigma_sqrt + scale * Sigma_diff
        Sigma      = torch.matmul(Sigma_sqrt, Sigma_sqrt.T)
        
       
    final_G_squared = torch.sum((mu - hat_mu)**2) + torch.sum((sqrtm_svd(Sigma) - hat_Sigma_sqrt)**2)
    if final_G_squared > epsilon**2 + tol:
        print(f"Warning: Final G_squared = {final_G_squared.item()} > {epsilon**2}")
    
    return mu, Sigma

def verify_gelbrich_constraint(mu, Sigma, hat_mu, hat_Sigma, radius):
    """
    Verify constraint
    """
    mu_dist_sq = torch.sum((mu - hat_mu)**2)
    # print(f"mu distance squared: {mu_dist_sq.item()}")
    
    Sigma_sqrt = sqrtm_svd(Sigma)
    hat_Sigma_sqrt = sqrtm_svd(hat_Sigma)
    Sigma_dist_sq = torch.sum((Sigma_sqrt - hat_Sigma_sqrt)**2)
    # print(f"Sigma distance squared: {Sigma_dist_sq.item()}")
    
    G_squared = mu_dist_sq + Sigma_dist_sq
    # print(f"Total G_squared: {G_squared.item()}, epsilon^2: {epsilon**2}")

    G_squared       = round(G_squared.item(), 5)
    radius_squared = round(radius**2, 5)
    
    return G_squared <= radius_squared, G_squared, radius_squared


def condition_number(matrix):
    """
    Computes the condition number of a matrix using the 2-norm.

    Parameters:
        matrix (np.ndarray): Input matrix (can be square or rectangular).

    Returns:
        float: The condition number of the matrix.
    """
    # Compute the singular values of the matrix
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    # Condition number is the ratio of the largest to smallest singular value
    cond_number = singular_values.max() / singular_values.min()

    return cond_number