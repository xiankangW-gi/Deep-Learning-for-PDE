import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


np.random.seed(42)
tf.random.set_seed(42)

def sin_activation(x):
    return tf.sin(x)

# ============================================================================
# FBSNN BASE CLASS (TensorFlow 2) 
# ============================================================================

class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers):

        self.Xi = Xi # initial point
        self.T = T # terminal time

        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions

        # layers
        self.layers = layers # (D+1) --> 1

        
        self.model = self.build_model(layers)

       
        self.optimizer = tf.keras.optimizers.Adam()

        
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )

    def build_model(self, layers):
        
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Dense(
            layers[1],
            input_dim=layers[0],
            activation=sin_activation,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ))

        # Hidden layers
        for i in range(2, len(layers)-1):
            model.add(tf.keras.layers.Dense(
                layers[i],
                activation=sin_activation,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'
            ))

        # Output layer
        model.add(tf.keras.layers.Dense(
            layers[-1],
            activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ))

        return model

    def net_u(self, t, X): # M x 1, M x D
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            inputs = tf.concat([t, X], 1)
            u = self.model(inputs, training=True) # M x 1

        Du = tape.gradient(u, X) # M x D
        del tape

        if Du is None:
            Du = tf.zeros_like(X)

        return u, Du

    def Dg_tf(self, X): # M x D
        
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)

        Dg = tape.gradient(g, X)
        if Dg is None:
            Dg = tf.zeros_like(X)

        return Dg



    @tf.function
    def loss_function(self, t, W, Xi):  # M x (N+1) x 1, M x (N+1) x D, 1 x D
        dt = self.T / self.N
        loss = 0.0
        
        
        t_curr, W_curr = t[:, 0, :], W[:, 0, :]
        X_curr = tf.tile(Xi, [self.M, 1])  # M x D
        Y_curr, Z_curr = self.net_u(t_curr, X_curr)  # M x 1, M x D
        
        X_list = [X_curr]
        Y_list = [Y_curr]
        
        
        for n in range(self.N):
            t_next, W_next = t[:, n+1, :], W[:, n+1, :]
            dW = W_next - W_curr  # M x D
            
            
            mu_term = self.mu_tf(t_curr, X_curr, Y_curr, Z_curr) * dt
            sigma_matrix = self.sigma_tf(t_curr, X_curr, Y_curr)  # M x D x D
            sigma_term = tf.linalg.matvec(sigma_matrix, dW)  # M x D
            X_next = X_curr + mu_term + sigma_term
            
          
            phi_term = self.phi_tf(t_curr, X_curr, Y_curr, Z_curr) * dt
            Z_sigma_term = tf.reduce_sum(Z_curr * sigma_term, axis=1, keepdims=True)
            Y_tilde = Y_curr + phi_term + Z_sigma_term
            
            Y_next, Z_next = self.net_u(t_next, X_next)
            
           
            loss += tf.reduce_mean(tf.square(Y_next - Y_tilde))
            
           
            t_curr, W_curr, X_curr, Y_curr, Z_curr = t_next, W_next, X_next, Y_next, Z_next
            X_list.append(X_curr)
            Y_list.append(Y_curr)
        
        
        loss += tf.reduce_mean(tf.square(Y_curr - self.g_tf(X_curr)))
        loss += tf.reduce_mean(tf.square(Z_curr - self.Dg_tf(X_curr)))
        
        
        X = tf.stack(X_list, axis=1)
        Y = tf.stack(Y_list, axis=1)
        
        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):

      T, M, N, D = self.T, self.M, self.N, self.D

      dt = T / N

      
      dt_array = np.zeros((M, N+1, 1), dtype=np.float32)
      dt_array[:,1:,:] = dt

      
      DW = np.zeros((M, N+1, D), dtype=np.float32)
      DW[:,1:,:] = np.sqrt(dt) * np.random.normal(size=(M, N, D)).astype(np.float32)

      
      t = np.cumsum(dt_array, axis=1)  #  (M, N+1, 1)
      W = np.cumsum(DW, axis=1)        #  (M, N+1, D)

      return t, W


    @tf.function
    def train_step(self, t_batch, W_batch, Xi):
        with tf.GradientTape() as tape:
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, Xi)

            # Add L2 regularization
            l2_loss = tf.add_n([
                tf.nn.l2_loss(w) for w in self.model.trainable_weights
            ])
            total_loss = loss + 1e-5 * l2_loss

        
        gradients = tape.gradient(total_loss, self.model.trainable_weights)

        
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)

       
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss, Y0_pred, grad_norm

    def train(self, N_Iter, learning_rate):
        
        self.optimizer.learning_rate.assign(learning_rate)

        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0

        for it in range(N_Iter):

            t_batch, W_batch = self.fetch_minibatch()

            
            t_batch = tf.constant(t_batch, dtype=tf.float32)
            W_batch = tf.constant(W_batch, dtype=tf.float32)
            Xi = tf.constant(self.Xi, dtype=tf.float32)

            loss_value, Y0_value, grad_norm = self.train_step(t_batch, W_batch, Xi)

            
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1

            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Grad Norm: %.3e, Time: %.2f, LR: %.3e' %
                      (it, loss_value.numpy(), Y0_value.numpy(), grad_norm.numpy(),
                       elapsed, learning_rate))
                start_time = time.time()

            # Adaptive learning rate reduction
            if it > 0 and it % 5000 == 0:
                current_lr = self.optimizer.learning_rate.numpy()
                new_lr = current_lr * 0.8
                self.optimizer.learning_rate.assign(new_lr)
                print(f"Reduced learning rate to {new_lr:.3e}")

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = tf.constant(Xi_star, dtype=tf.float32)
        t_star = tf.constant(t_star, dtype=tf.float32)
        W_star = tf.constant(W_star, dtype=tf.float32)

        loss, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)

        return X_star.numpy(), Y_star.numpy()

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
        M = self.M
        D = self.D
        return tf.zeros([M,D], dtype=tf.float32)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = self.M
        D = self.D
        return tf.eye(D, batch_shape=[M], dtype=tf.float32)

# ============================================================================
# HAMILTON-JACOBI-BELLMAN EQUATION 
# ============================================================================

class HamiltonJacobiBellman(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers):
        super().__init__(Xi, T, M, N, D, layers)

    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return tf.reduce_sum(Z**2, 1, keepdims=True) # M x 1

    def g_tf(self, X): # M x D
        return tf.math.log(0.5 + 0.5*tf.reduce_sum(X**2, 1, keepdims=True)) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D (zeros)

    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return tf.sqrt(2.0) * super().sigma_tf(t, X, Y) # M x D x D

# ============================================================================
# MAIN EXECUTION 
# ============================================================================

if __name__ == "__main__":

    print("Starting improved 100-dimensional Hamilton-Jacobi-Bellman equation solver...")
    print("Using TensorFlow", tf.__version__)

   
    M = 100 # number of trajectories (batch size) 
    N = 50 # number of time snapshots 
    D = 100 # number of dimensions

    # Improved network architecture
    layers = [D+1] + 4*[256] + [1]  
    Xi = np.zeros([1,D])
    T = 1.0

   
    print("\nInitializing model...")
    model = HamiltonJacobiBellman(Xi, T, M, N, D, layers)

    total_params = sum([tf.size(var).numpy() for var in model.model.trainable_weights])
    print(f"Model architecture: {layers}")
    print(f"Total parameters: {total_params}")

    
    print("\nPhase 1: Initial training (LR: 1e-3)")
    model.train(N_Iter=5000, learning_rate=1e-3)

    print("\nPhase 2: Fine-tuning (LR: 5e-4)")
    model.train(N_Iter=5000, learning_rate=5e-4)

    print("\nPhase 3: Final refinement (LR: 1e-4)")
    model.train(N_Iter=5000, learning_rate=1e-4)

    print("\nTraining completed! Generating test results...")

   
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    def u_exact(t, X): 
        MC = 10000  
        NC = t.shape[0]

        
        W = np.random.normal(size=(MC, NC, D))

        
        X_expanded = np.expand_dims(X, 0)  # 1 x NC x D
        t_expanded = np.expand_dims(t.squeeze(), 0)  # 1 x NC

        
        sqrt_term = np.sqrt(2.0 * np.abs(T - t_expanded))  # 1 x NC
        sqrt_term = np.expand_dims(sqrt_term, -1)  # 1 x NC x 1

        
        X_terminal = X_expanded + sqrt_term * W

        
        X_squared = np.sum(X_terminal**2, axis=2, keepdims=True)  # MC x NC x 1
        g_values = np.log(0.5 + 0.5 * X_squared)

        
        u_values = -np.log(np.mean(np.exp(-g_values), axis=0))  # NC x 1

        return u_values.squeeze()  # NC

    print("Computing exact solution for comparison...")
    Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:])

   
    Y_test_terminal = np.log(0.5 + 0.5*np.sum(X_pred[:,-1,:]**2, axis=1, keepdims=True))

    
    print(f"\nResults after training:")
    print(f"Initial value Y0 (learned): {Y_pred[0,0,0]:.6f}")
    print(f"Initial value Y0 (exact): {Y_test[0]:.6f}")
    initial_error = abs(Y_pred[0,0,0] - Y_test[0])
    relative_error = initial_error / abs(Y_test[0])
    print(f"Absolute error: {initial_error:.6f}")
    print(f"Relative error: {relative_error:.6f} ({relative_error*100:.3f}%)")


    plt.figure(figsize=(12, 8))

    
    plt.plot(t_test[0:1,:,0].T, Y_pred[0:1,:,0].T, 'b-', linewidth=2, label='Learned $u(t,X_t)$')

    plt.plot(t_test[0,:,0].T, Y_test, 'r--', linewidth=2, label='Exact $u(t,X_t)$')

    
    plt.plot(t_test[0:1,-1,0], Y_test_terminal[0:1,0], 'ks', markersize=8,
             label='$Y_T = u(T,X_T)$')

    
    plt.plot([0], Y_test[0], 'ko', markersize=8, label='$Y_0 = u(0,X_0)')

    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$Y_t = u(t,X_t)$', fontsize=14)
    plt.title('100-dimensional Hamilton-Jacobi-Bellman Equation Solution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

   
    errors = np.abs(Y_test - Y_pred[0,:,0]) / np.abs(Y_test)

    plt.figure(figsize=(12, 6))
    plt.plot(t_test[0,:,0], errors, 'b-', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title('100-dimensional Hamilton-Jacobi-Bellman: Relative Error Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    print(f"Max relative error: {np.max(errors):.6f}")
    print(f"Mean relative error: {np.mean(errors):.6f}")
