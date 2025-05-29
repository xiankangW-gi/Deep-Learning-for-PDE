import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def sin_activation(x):
    return torch.sin(x)

# ============================================================================
# NEURAL NETWORK MODULE
# ============================================================================

class FBSNNModel(nn.Module):
    def __init__(self, layers):
        super(FBSNNModel, self).__init__()
        
        self.layers_list = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i+1])
            # Initialize weights
            nn.init.xavier_uniform_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            self.layers_list.append(linear_layer)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            # Apply sin activation to all layers except the last one
            if i < len(self.layers_list) - 1:
                x = sin_activation(x)
        return x

# ============================================================================
# FBSNN BASE CLASS (PyTorch) - IMPROVED
# ============================================================================

class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers):
        self.Xi = Xi  # initial point
        self.T = T    # terminal time
        self.M = M    # number of trajectories
        self.N = N    # number of time snapshots
        self.D = D    # number of dimensions
        self.layers = layers  # (D+1) --> 1
        
        # Build the model
        self.model = FBSNNModel(layers).to(device)
        
        # Optimizer with better learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def net_u(self, t, X):
        """Compute u and Du with improved gradient computation"""
        # Ensure X requires gradients
        # This X needs to be a leaf node for autograd.grad to compute gradients w.r.t. X.
        # .clone().detach().requires_grad_(True) makes it a new leaf.
        X_leaf = X.clone().detach().requires_grad_(True)
        
        # Ensure t has the right shape: if t is M x 1 x 1, squeeze to M x 1
        if t.dim() == 3:
            t_squeezed = t.squeeze(-1)  # M x 1
        else:
            t_squeezed = t # Assuming t is already M x 1
        
        # Concatenate t and X_leaf
        inputs = torch.cat([t_squeezed, X_leaf], dim=1)  # M x (D+1)
        
        # Forward pass
        u = self.model(inputs)  # M x 1
        
        # Compute gradients
        # create_graph=True is essential because Z (which is Du) is part of the BSDE dynamics,
        # and the overall loss needs to be backpropagated through Z.
        # retain_graph will be True by default if create_graph is True.
        Du = torch.autograd.grad(
            outputs=u,
            inputs=X_leaf, # Differentiate w.r.t. the leaf version of X
            grad_outputs=torch.ones_like(u),
            create_graph=True, 
            retain_graph=True, # Needed as create_graph=True
            only_inputs=True # Ensures grads are only w.r.t. X_leaf among inputs to grad()
        )[0]  # M x D
        
        return u, Du

    def Dg_tf(self, X):
        """Compute gradient of terminal condition"""
        X_leaf = X.clone().detach().requires_grad_(True)
        g = self.g_tf(X_leaf) # Pass the leaf tensor to g_tf
        
        Dg = torch.autograd.grad(
            outputs=g,
            inputs=X_leaf,
            grad_outputs=torch.ones_like(g),
            create_graph=True, 
            retain_graph=True, # Needed as create_graph=True
            only_inputs=True
        )[0]
        
        return Dg

    def loss_function(self, t, W, Xi_input_tensor): # Renamed Xi to avoid conflict with self.Xi
        """Improved loss function with better numerical stability"""
        loss = 0.0
        X_list = []
        Y_list = []
        
        # Fix dimension handling for t and W
        t0 = t[:, 0, :].squeeze(-1).unsqueeze(-1)  # M x 1
        W0 = W[:, 0, :]    # M x D
        X0 = Xi_input_tensor.repeat(self.M, 1)  # M x D, use the passed Xi
        Y0, Z0 = self.net_u(t0, X0)  # M x 1, M x D
        
        X_list.append(X0)
        Y_list.append(Y0)
        
        dt = self.T / self.N
        
        for n in range(self.N):
            t1 = t[:, n+1, :].squeeze(-1).unsqueeze(-1)  # M x 1
            W1 = W[:, n+1, :]      # M x D
            
            # Compute X1 with improved numerical stability
            mu_term = self.mu_tf(t0, X0, Y0, Z0) * dt
            
            # Improved sigma computation
            sigma_matrix = self.sigma_tf(t0, X0, Y0)  # M x D x D
            dW = W1 - W0  # M x D
            
            # Matrix multiplication for sigma * dW
            sigma_term = torch.bmm(sigma_matrix, dW.unsqueeze(-1)).squeeze(-1)  # M x D
            
            X1 = X0 + mu_term + sigma_term # X1's graph depends on Y0, Z0
            
            # Compute Y1_tilde
            phi_term = self.phi_tf(t0, X0, Y0, Z0) * dt
            Z_sigma_term = torch.sum(Z0 * sigma_term, dim=1, keepdim=True)
            Y1_tilde = Y0 + phi_term + Z_sigma_term
            
            Y1, Z1 = self.net_u(t1, X1) # X1 is not a leaf here, but net_u handles it
            
            # Add step loss
            step_loss = torch.mean((Y1 - Y1_tilde) ** 2)
            loss += step_loss
            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            
            X_list.append(X0)
            Y_list.append(Y0)
        
        # Terminal condition losses
        terminal_loss_Y = torch.mean((Y1 - self.g_tf(X1)) ** 2)
        # Z1 comes from net_u(t1, X1), Dg_tf(X1) is the target.
        # Both will have graphs if called appropriately.
        terminal_loss_Z = torch.mean((Z1 - self.Dg_tf(X1)) ** 2) 
        
        loss += terminal_loss_Y
        loss += terminal_loss_Z
        
        # These are for returning the path, not directly for loss minimization here
        X_stacked = torch.stack(X_list, dim=1)  # M x (N+1) x D
        Y_stacked = torch.stack(Y_list, dim=1)  # M x (N+1) x 1
        
        return loss, X_stacked, Y_stacked, Y_stacked[0, 0, 0]


    def fetch_minibatch(self):
        """Generate random batch of Brownian motion paths"""
        T, M, N, D = self.T, self.M, self.N, self.D
        dt = T / N
        
        # Time steps
        dt_array = np.zeros((M, N+1, 1), dtype=np.float32)
        dt_array[:, 1:, :] = dt
        
        # Brownian increments
        DW = np.zeros((M, N+1, D), dtype=np.float32)
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D)).astype(np.float32)
        
        # Construct t, W
        t = np.cumsum(dt_array, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)        # M x (N+1) x D
        
        return t, W

    def train_step(self, t_batch, W_batch, Xi_tensor): # Renamed Xi to Xi_tensor
        """Optimized training step with better gradient handling"""
        self.optimizer.zero_grad()
        
        loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, Xi_tensor)
        
        # Add L2 regularization
        l2_reg_loss = 0.0
        for param in self.model.parameters():
            l2_reg_loss += torch.norm(param) ** 2 
        total_loss = loss + 1e-5 * l2_reg_loss
        
        # Backward pass
        total_loss.backward() # This is where create_graph=True in net_u becomes crucial
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item(), Y0_pred.item(), grad_norm.item()

    def train(self, N_Iter, learning_rate):
        """Improved training with better monitoring"""
        # Set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        start_time = time.time()
        # best_loss = float('inf') # Not used
        
        # Convert self.Xi (numpy array) to a tensor once
        Xi_tensor = torch.tensor(self.Xi, dtype=torch.float32, device=device)

        for it in range(N_Iter):
            t_batch_np, W_batch_np = self.fetch_minibatch()
            
            # Convert to PyTorch tensors for this batch
            t_batch = torch.tensor(t_batch_np, dtype=torch.float32, device=device)
            W_batch = torch.tensor(W_batch_np, dtype=torch.float32, device=device)
            
            loss_value, Y0_value, grad_norm = self.train_step(t_batch, W_batch, Xi_tensor) # Pass Xi_tensor
            
            # Update best loss - not strictly needed if not saving best model here
            # if loss_value < best_loss:
            #     best_loss = loss_value
            
            # Print progress
            if it % 100 == 0:
                elapsed = time.time() - start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                print('It: %d, Loss: %.3e, Y0: %.3f, Grad Norm: %.3e, Time: %.2f, LR: %.3e' %
                      (it, loss_value, Y0_value, grad_norm, elapsed, current_lr))
                start_time = time.time()
            
            # Learning rate scheduling
            if it > 0 and it % 1000 == 0: # Scheduler is typically called per epoch, but here per N iterations
                self.scheduler.step()

    def predict(self, Xi_star_np, t_star_np, W_star_np): # Suffix _np for numpy inputs
        """Prediction function"""
        self.model.eval()  # Set model to evaluation mode (e.g., for dropout, batchnorm)

        # Convert numpy inputs to PyTorch tensors
        Xi_star_tensor = torch.tensor(Xi_star_np, dtype=torch.float32, device=device)
        t_star_tensor = torch.tensor(t_star_np, dtype=torch.float32, device=device)
        W_star_tensor = torch.tensor(W_star_np, dtype=torch.float32, device=device)
            
        # Perform the forward pass and simulation steps.
        # The loss_function calls net_u, which computes Du (Z). These computations
        # require gradient tracking to be enabled (which is default).
        # We are NOT using `with torch.no_grad():` here because net_u needs to compute grads.
        # Since we don't call .backward() on the loss, model parameters are not updated.
        _, X_star, Y_star, _ = self.loss_function(t_star_tensor, W_star_tensor, Xi_star_tensor)
        
        self.model.train()  # Set model back to training mode for subsequent training phases
        return X_star.detach().cpu().numpy(), Y_star.detach().cpu().numpy()


    ###########################################################################
    ############################# Abstract Methods ###########################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        # M = self.M # M is not always self.M if called from predict with different batch size
        M = X.shape[0]
        D = self.D
        return torch.zeros(M, D, dtype=torch.float32, device=device)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        # M = self.M
        M = X.shape[0]
        D = self.D
        return torch.eye(D, device=device).unsqueeze(0).repeat(M, 1, 1)

# ============================================================================
# HAMILTON-JACOBI-BELLMAN EQUATION SOLVER - IMPROVED
# ============================================================================

class HamiltonJacobiBellman(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers):
        super().__init__(Xi, T, M, N, D, layers)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return torch.sum(Z ** 2, dim=1, keepdim=True)  # M x 1

    def g_tf(self, X):  # M x D
        return torch.log(0.5 + 0.5 * torch.sum(X ** 2, dim=1, keepdim=True))  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D (zeros)

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        # Ensure super call gets correct M
        M = X.shape[0] # Get M from X's current batch size
        # Temporarily set self.M for super().sigma_tf if it strictly uses self.M
        # Or better, ensure base methods also use X.shape[0]
        # The provided base class already uses X.shape[0] via self.D which is okay for D.
        # For M, it's better to pass it or derive from X.shape[0].
        # Current base sigma_tf and mu_tf were updated to use X.shape[0] for M.
        return torch.sqrt(torch.tensor(2.0, device=device)) * super().sigma_tf(t, X, Y)


# ============================================================================
# MAIN EXECUTION - IMPROVED WITH FIXED PLOTTING
# ============================================================================

if __name__ == "__main__":
    print("Starting improved 100-dimensional Hamilton-Jacobi-Bellman equation solver...")
    print("Using PyTorch", torch.__version__)
    
    # Parameters - optimized for better convergence
    M_train = 100  # number of trajectories for training (batch size)
    N = 50   # number of time snapshots
    D = 100  # number of dimensions
    
    # Improved network architecture
    layers = [D+1] + 4*[256] + [1]
    Xi_np = np.zeros([1, D], dtype=np.float32) # Ensure float32 for consistency
    T = 1.0
    
    # Training with improved strategy
    print("\nInitializing model...")
    # Pass M_train as the batch size for training
    model = HamiltonJacobiBellman(Xi_np, T, M_train, N, D, layers)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Model architecture: {layers}")
    print(f"Total parameters: {total_params}")
    
    # Multi-stage training with better learning rates
    print("\nPhase 1: Initial training (LR: 1e-3)")
    model.train(N_Iter=5000, learning_rate=1e-3)
    
    print("\nPhase 2: Fine-tuning (LR: 5e-4)")
    model.train(N_Iter=5000, learning_rate=5e-4)
    
    print("\nPhase 3: Final refinement (LR: 1e-4)")
    model.train(N_Iter=5000, learning_rate=1e-4)
    
    print("\nTraining completed! Generating test results...")
    
    # Testing with exact solution
    # For testing, we can use a different M if desired, e.g. M_test.
    # Here, fetch_minibatch uses self.M (M_train). If we want M_test for prediction,
    # we'd need a way to generate t_test, W_test with M_test samples.
    # For simplicity, using M_train for test batch generation.
    t_test_np, W_test_np = model.fetch_minibatch() # Fetches M_train samples
    
    # model.predict expects numpy arrays for Xi, t, W
    X_pred_np, Y_pred_np = model.predict(Xi_np, t_test_np, W_test_np)
    
    def u_exact(t_np, X_np_paths):  # Exact solution using Monte Carlo
        # t_np: N+1 array of time points for one trajectory (e.g., t_test_np[0, :, 0])
        # X_np_paths: M x N+1 x D array of state paths (e.g., X_pred_np)
        # We want to compute u_exact(t_k, X_k) for one sample path, e.g., path 0.
        # So, t_np is (N+1,) and X_current_path is (N+1, D)
        
        MC = 10000  # Monte Carlo samples
        num_time_points = t_np.shape[0] # Should be N+1
        
        u_values_path = np.zeros(num_time_points)

        for k in range(num_time_points): # Iterate over each time point in the path
            tk = t_np[k]
            Xk = X_np_paths[k, :] # D-dimensional state at time tk

            if tk >= T: # If at or beyond terminal time
                # Terminal condition: log(0.5 + 0.5*|X_T|^2)
                Xk_squared_sum = np.sum(Xk**2)
                u_values_path[k] = np.log(0.5 + 0.5 * Xk_squared_sum)
                continue

            # Generate random samples for this specific (tk, Xk)
            # W_mc is (MC, D) for Wiener process increments from tk to T
            W_mc = np.random.normal(size=(MC, D)) 
            
            # Time-dependent term: sqrt(2 * (T - tk))
            sqrt_term_mc = np.sqrt(2.0 * (T - tk))
            
            # Compute terminal positions: X_T = X_k + sqrt(2*(T-tk)) * W
            # Xk needs to be (1, D) for broadcasting with W_mc (MC, D)
            X_terminal_mc = Xk.reshape(1, D) + sqrt_term_mc * W_mc # MC x D
            
            # Terminal condition values for MC samples: log(0.5 + 0.5*|X_T|^2)
            X_terminal_mc_squared_sum = np.sum(X_terminal_mc**2, axis=1) # MC,
            g_values_mc = np.log(0.5 + 0.5 * X_terminal_mc_squared_sum) # MC,
            
            # Cole-Hopf: u(t,x) = -log(E[exp(-g(X_T))])
            # We need E[exp(-g(X_T)) | X_tk = Xk]
            # Note: The problem setup implies u(t,x) = - (1/lambda) log E[exp(-lambda g(X_T))]
            # Here, lambda is implicitly 1 based on the phi_tf = sum(Z^2) which corresponds to lambda=1/2 for HJB u_t + 1/2 Tr(sigma sigma^T D^2 u) + H(x, Du) = 0
            # The specific HJB is u_t + Delta u + |Du|^2 = 0 with g(x) = ln(1/2 + 1/2 |x|^2) and X_t = sqrt(2) B_t
            # Its Feynman-Kac solution is u(t,x) = -log E[exp(-g(x + sqrt(2(T-t)) W_1))]
            # This matches the form used in the original problem from which this code is derived.
            
            mean_exp_neg_g = np.mean(np.exp(-g_values_mc))
            u_values_path[k] = -np.log(mean_exp_neg_g)
            
        return u_values_path # (N+1,) array of exact u values for the input path

    print("Computing exact solution for comparison (for the first predicted trajectory)...")
    # Get the time points for the first trajectory
    t_points_for_exact = t_test_np[0, :, 0] # Shape (N+1,)
    # Get the state path for the first trajectory
    X_path_for_exact = X_pred_np[0, :, :]   # Shape (N+1, D)
    
    Y_test_exact_path0 = u_exact(t_points_for_exact, X_path_for_exact) # Shape (N+1,)
    
    # Compute terminal condition values for all predicted paths at terminal time
    Y_pred_terminal_values = np.log(0.5 + 0.5*np.sum(X_pred_np[:, -1, :]**2, axis=1, keepdims=True)) # M x 1
    
    # Print results
    print(f"\nResults after training (for trajectory 0):")
    print(f"Initial value Y0 (learned): {Y_pred_np[0, 0, 0]:.6f}")
    print(f"Initial value Y0 (exact for X0=Xi): {Y_test_exact_path0[0]:.6f}")
    initial_error = abs(Y_pred_np[0, 0, 0] - Y_test_exact_path0[0])
    # Avoid division by zero if Y_test_exact_path0[0] is close to zero
    relative_error_denom = abs(Y_test_exact_path0[0]) if abs(Y_test_exact_path0[0]) > 1e-9 else 1.0
    relative_error = initial_error / relative_error_denom
    print(f"Absolute error at t=0: {initial_error:.6f}")
    print(f"Relative error at t=0: {relative_error:.6f} ({relative_error*100:.3f}%)")
    
    # Enhanced plotting with initial and terminal conditions marked
    print("Generating enhanced plots...")
    
    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style

    # Plot 1: Main solution comparison (for trajectory 0)
    plt.figure(figsize=(12, 8))
    
    plt.plot(t_points_for_exact, Y_pred_np[0, :, 0], 'b-', linewidth=2, label='Learned $u(t,X_t)$ (Path 0)')
    plt.plot(t_points_for_exact, Y_test_exact_path0, 'r--', linewidth=2, label='Exact $u(t,X_t)$ (Path 0, via MC)')
    
    plt.plot(t_points_for_exact[-1], Y_pred_terminal_values[0, 0], 'bs', markersize=8, alpha=0.7,
             label='Learned $Y_T = u(T,X_T)$ (Path 0)')
    plt.plot(t_points_for_exact[-1], Y_test_exact_path0[-1], 'rs', markersize=8, fillstyle='none',
             label='Exact $Y_T = u(T,X_T)$ (Path 0, via MC)')
             
    plt.plot(t_points_for_exact[0], Y_pred_np[0,0,0], 'bo', markersize=8, alpha=0.7,
             label='Learned $Y_0 = u(0,X_0)$')
    plt.plot(t_points_for_exact[0], Y_test_exact_path0[0], 'ro', markersize=8, fillstyle='none',
             label='Exact $Y_0 = u(0,X_0)$ (via MC)')
    
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$Y_t = u(t,X_t)$', fontsize=14)
    plt.title(f'{D}-dimensional Hamilton-Jacobi-Bellman Equation Solution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('solution_comparison.png')
    
    # Plot 2: Error analysis (for trajectory 0)
    errors_path0 = np.abs(Y_test_exact_path0 - Y_pred_np[0, :, 0])
    relative_errors_path0_denom = np.abs(Y_test_exact_path0)
    relative_errors_path0_denom[relative_errors_path0_denom < 1e-9] = 1.0 # Avoid division by zero
    relative_errors_path0 = errors_path0 / relative_errors_path0_denom
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_points_for_exact, relative_errors_path0, 'b-', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('Relative Error (Path 0)', fontsize=14)
    plt.title(f'{D}-dim HJB: Relative Error Over Time (Path 0)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('error_analysis.png')
    
    print(f"Max relative error (Path 0): {np.max(relative_errors_path0):.6f}")
    print(f"Mean relative error (Path 0): {np.mean(relative_errors_path0):.6f}")
    
    # Plot 3: Multiple trajectories analysis
    plt.figure(figsize=(12, 8))
    
    num_paths_to_plot = min(5, M_train)
    for i in range(num_paths_to_plot):
        label_learned = 'Learned trajectories' if i == 0 else None
        plt.plot(t_test_np[i, :, 0], Y_pred_np[i, :, 0], color='royalblue', alpha=0.7, linewidth=1.5, label=label_learned)
        
        # Mark terminal conditions for these learned trajectories
        label_terminal = '$Y_T = u(T,X_T)$ (Learned)' if i == 0 else None
        plt.plot(t_test_np[i, -1, 0], Y_pred_terminal_values[i, 0], 'ks', markersize=6, alpha=0.8, label=label_terminal)

    # Plot exact solution for path 0 as a reference
    plt.plot(t_points_for_exact, Y_test_exact_path0, 'r--', linewidth=3, label='Exact $u(t,X_t)$ (Path 0, via MC)')
    
    # Mark initial exact condition
    plt.plot(t_points_for_exact[0], Y_test_exact_path0[0], 'ro', markersize=10, fillstyle='none', label='Exact $Y_0 = u(0,X_0)$ (Path 0, via MC)')
    
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$Y_t = u(t,X_t)$', fontsize=14)
    plt.title(f'{D}-dim HJB: Multiple Trajectories & Boundary Conditions', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('multiple_trajectories.png')
    
    print("\nImproved PyTorch solver with enhanced visualization completed successfully!")