"""
Pendulum 2-nd-order ODE solved with PI-DEEPOnet
==============================================

Overview
--------
This script trains a Physics-Informed DEEPOnet that maps an input torque
function τ(t), defined on the interval [0, T], to the resulting pendulum
trajectory θ(t) and θ̇(t) under given mechanical properties.

Data generation
---------------
- Collocation points (the time coordinates where the ODE residual is enforced)
  are drawn from a Sobol low-discrepancy sequence.
- Torque time functions are sampled from a Gaussian Random Field (GRF).

Key parameters
--------------
- m        : number of sensor points used to discretise the input torque.
- u        : vector of torque values evaluated at the sensor points.
- u_t      : torque profile interpolated on the collocation times
             (needed inside `pde_loss`).
- optimizer: Adam for the first 60 % of epochs, followed by L-BFGS.
- Validation loss is computed in a physics-informed manner, not with extra
  labelled data.

At the end of training the script plots an example trajectory to visualise the
learned solution.
"""

import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from scipy.ndimage import gaussian_filter


class DEEPONet(nn.Module): 
    def __init__(self, branch_dim, trunk_dim, hidden_dim, number_of_hidden_layers): 
        super(DEEPONet, self).__init__()
        # Define sub-network for first input stream 
        
        activation_fn=nn.Tanh()
        
        layer_width = branch_dim
        layers = []
        for n_layer in range(number_of_hidden_layers):
            layers.append(nn.Linear(layer_width,hidden_dim))
            layers.append(activation_fn)
            layer_width = hidden_dim
        
        self.branch = nn.Sequential(*layers)
        
        layer_width = trunk_dim
        layers = []
        for n_layer in range(number_of_hidden_layers):
           layers.append(nn.Linear(layer_width,hidden_dim))
           layers.append(activation_fn)
           layer_width = hidden_dim
       
        self.trunk = nn.Sequential(*layers)
        
        for layer in self.branch:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
                
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        
   
    def forward(self, time, torque): # Process each input stream 
        out_branch = self.branch(torque) # Output from the branch net (takes the torque function as input) 
        out_trunk = self.trunk(time)     # Output from the trunk net (takes the coordinate as input) 
        # Concatenate outputs from both branches 
        dot_product = torch.mul(out_branch,out_trunk)
        output = torch.sum(dot_product,2) 
        
        ## It is possible to add also a bias weigth here
    
        return torch.unsqueeze(output,2)


class PINN():
    def __init__(self,NN_model,lr):
        self.network = NN_model
    
    def loss_pde(self, X):
        
        t = X[:,:,:1].requires_grad_()
        u = X[:,:,1:1+m] # Torque fed to the NET 
        u_t = X[:,:,-1:] # Torque evaluated at time coordinate t

        y = self.network(t,u)
        ## .grad() returns a tuple: the gradient is teh first element --> [0]
        y_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph = True)[0]
        ## when the ouput is a vector, you must define grad_outputs=torch.ones_like(y)
        y_tt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph = True)[0]

        loss = y_tt + y_t*c/J + y*k/J + g/L*torch.sin(y) -u_t/J
        
        return torch.mean(loss**2)
    
    def loss_ic(self, X):
        
        u = X[:,:1,1:1+m] # Torque fed to the NET 
        t = torch.zeros(u.shape)[:,:,:1].requires_grad_()
        
        y = self.network(t,u)
        y_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph = True)[0]
        
        loss1 = torch.mean(y**2)
        loss2 = torch.mean(y_t**2)
        
        return loss1 + loss2
    
    def train(self, num_epochs, lr, train_loader, Xval, only_LBFGS, saving_name):
        
        if only_LBFGS:
            self.optimizer = torch.optim.LBFGS(self.network.parameters(),
                                history_size=10, 
                                max_iter=8, 
                                line_search_fn="strong_wolfe")
        else: 
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.network.train()
        best_loss = float('inf'); loss_tracker = []; val_loss_tracker = []
        
        
        def closure():
            self.optimizer.zero_grad()
            loss_pde = self.loss_pde(X)
            loss_ic = self.loss_ic(X)
            objective = loss_pde + loss_ic
            objective.backward()
            return objective
        
        for epoch in range(num_epochs):
            
            epoch_loss = 0
            
            for X in train_loader:
            
                self.optimizer.step(closure)
                loss = closure()
                epoch_loss += loss.item()
            
            loss_tracker.append(epoch_loss)
            
            val_loss = self.loss_pde(Xval) + self.loss_ic(Xval)
            val_loss_tracker.append(val_loss.item())
            
            if epoch % 25 == 0:
                print('Epoch {}/{}, Training loss = {:10.2e}, Validation loss = {:10.2e}'.format(epoch, num_epochs, epoch_loss, val_loss.item()))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.network, saving_name)

            if epoch == int(0.6 * num_epochs) and not only_LBFGS: 
                self.optimizer = torch.optim.LBFGS(self.network.parameters(),
                                    history_size=10, 
                                    max_iter=8, 
                                    line_search_fn="strong_wolfe")
                print('Optimzer switch') 
        
        fig, ax1 = plt.subplots(1,1,figsize=(6,4))
        ax1.set_yscale('log')
        ax1.plot(loss_tracker,'b',label='Training loss')
        ax1.plot(val_loss_tracker,'orange',label='Validation loss')
        ax1.set_title('Training loss')
        ax1.legend()
        ax1.grid(visible=True)

    def predict(self, T,U): 
        t = torch.Tensor(T.reshape(1,-1,1))
        u = torch.Tensor(U.reshape(1,1,-1))
        y = self.network(t,u)
        return y.cpu().detach().numpy()[0,:,0]

def generate_grf(n, scale_length=10):
    noise = np.random.normal(0, 1, (n,))
    grf = gaussian_filter(noise, sigma=scale_length)

    # Normalize the GRF
    grf -= np.mean(grf)
    grf /= np.std(grf)

    return grf*10 # scale the torque between -10 and 10 (more or less)

def data(N, Ny, num_batches, device = torch.device("cpu"), validation = False):
    """
    
    Parameters
    ----------
    N : Number of different training input functions
    Ny : Number of training collocation points (coordinates where to train the NN)
    num_batches : Number of batches
    validation : condition for generating the validation dataset (no batches) 
    device : 'cpu' or 'gpu'

    Returns
    -------
    data_loader : the dataset used for training

    """
    X_pde = np.zeros([N, Ny, 1 + m + 1])
    soboleng_pde = torch.quasirandom.SobolEngine(dimension=1)
    coll_points_pde =  soboleng_pde.draw(Ny).detach().numpy()*T ## Coordinates in range [0,T]
    X_pde[:,:,:1] = coll_points_pde
    
    sensor_points = np.linspace(0,T,m)
    
    for idx in range(N):
        u_0 = generate_grf(m)
        u_x = np.interp(coll_points_pde[:,0], sensor_points, u_0)
        X_pde[idx,:,1:1+m] = u_0
        X_pde[idx,:,-1] = u_x

    data_loader = np.array_split(X_pde, num_batches, axis=0)
    
    if validation:
        data_loader = X_pde
    
    return torch.tensor(data_loader, dtype=torch.float32, device = device) 


########## ########## ########## ##########
## Mechanical parameters 
k = torch.tensor(1, dtype=torch.float32) # Stiffness 
c = torch.tensor(0.5, dtype=torch.float32) # Nondimensional Damping 
m = torch.tensor(1, dtype=torch.float32) # Conncentrated mass 
L = torch.tensor(5, dtype=torch.float32) # Length of pendulum (Inertia = mL^2)
T = 10 # Simualtion time
J = m*L**2 # Inertia 
g = torch.tensor(9.81, dtype=torch.float32) # Gravitational acceleration 


m = 50  ## Dimensions of the branch net
t = 1

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

new_model = True
training = True

if new_model:
    net = DEEPONet(branch_dim=m,
                  trunk_dim=t, hidden_dim=50, 
                  number_of_hidden_layers=4)
else:
    net = torch.load('model.pth', weights_only=False)

model = PINN(net,1e-4) 

if training:
    train_loader = data(100,1001,1, device)
    Xval = data(10,101,1, device, validation = True) # num_batches is not considered if validation = True
    pinn = model.train(num_epochs=1000, lr = 1e-4, train_loader = train_loader , Xval = Xval, only_LBFGS = False, saving_name = 'model.pth')
    
########################################################

## PLANT simulation

k = k.numpy(); c = c.numpy(); J = J.numpy(); g = g.numpy(); L = L.numpy()

freq = 50   # Number of samples per second
time_step = 1/freq
t_num = T*freq
time = np.linspace(0,T,t_num+1)[:,None]
## ICs
x1 = np.array([0]); x2 = np.array([0]); u = generate_grf(50); y_true = []
u_t = np.interp(time, np.linspace(0,T,50), u)
y_sim = model.predict(time,u)

import math

for j in range(len(time)):
    y_true.append(x1)
    x1p =x1 + x2*time_step
    x2p = - x2*c/J - x1*k/J - g/L*math.sin(x1) + u_t[j]/J
    x1 = x1p; x2 = x2p

fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(7, 5), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# --- Sub-plot 1 : y_true & y_sim --------------------------------------------
ax1.plot(time, np.array(y_true)*180/np.pi, label="Reference", linewidth=2)
ax1.plot(time, y_sim*180/np.pi, label="PINN",  linestyle="--", linewidth=2)
ax1.set_ylabel("Angle (°)")
ax1.legend(loc="best")

# --- Sub-plot 2 : u_t --------------------------------------------------------
ax2.plot(time, u_t, color="tab:red", linewidth=2)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("u(t) (Nm)")

fig.tight_layout()
plt.show()








