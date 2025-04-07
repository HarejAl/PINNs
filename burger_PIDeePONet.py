"""
Burger's differential equation solved with PI-DeePOnet
    - Training input functions are generated with a Gaussian Random Field (GRF) 
    - Hybrid optimizer: Adam + LBFGS
    - Viscosity is set to 0.1: an implementation could be take it as an input function
"""

import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from scipy.ndimage import gaussian_filter

class DeePOnet(nn.Module): 
    def __init__(self, branch_dim, trunk_dim, hidden_dim, number_of_hidden_layers): 
        """
        Parameters
        ----------
        branch_dim : input dimension of the branch net = number of sensor points
        trunk_dim : input dimension of trunk net = number of coordinates
        hidden_dim : width of hidden layers
        number_of_hidden_layers :number opf hidden layers
        """
        
        super(DeePOnet, self).__init__()
        
        activation_fn=nn.Tanh()

        layer_width = branch_dim
        layers1 = []
        for n_layer in range(number_of_hidden_layers):
            layers1.append(nn.Linear(layer_width,hidden_dim))
            layers1.append(activation_fn)
            layer_width = hidden_dim
        
        self.branch1 = nn.Sequential(*layers1)
        
        layer_width = trunk_dim
        layers3 = []
        for n_layer in range(number_of_hidden_layers):
           layers3.append(nn.Linear(layer_width,hidden_dim))
           layers3.append(activation_fn)
           layer_width = hidden_dim
       
        self.trunk = nn.Sequential(*layers3) 
        
        for layer in self.branch1:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
                
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        
    def forward(self, coordinate, input_fun): # Process each input stream 
        out1 = self.branch1(input_fun) 
        out_trunk = self.trunk(coordinate) 
        # Concatenate outputs from both branches 
        dot_product = torch.mul(out1,out_trunk)
        output = torch.sum(dot_product,2) 
        return torch.unsqueeze(output,2)


class PINN():
    def __init__(self,NN_model):
        self.network = NN_model

    def makeNetwork(self, t, x, u_0):
        t_x = torch.cat((t,x),2) # 1 is the concatenating axis
        return self.network(t_x, u_0)    
    
    def loss_pde(self, X):
        t = X[:,:,:1].requires_grad_()
        x = X[:,:,1:2].requires_grad_()
        u_0 = X[:,:1,2:2+b1]

        s = self.makeNetwork(t,x,u_0)
        s_t = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(s),  create_graph=True)[0]
        s_x = torch.autograd.grad(s, x, grad_outputs=torch.ones_like(s),  create_graph=True)[0]
        s_xx = torch.autograd.grad(s_x, x, grad_outputs=torch.ones_like(s),  create_graph=True)[0]
        
        loss = s_t + s*s_x - viscosity*s_xx 

        return torch.mean(loss**2)
    
    def loss_ic(self, XX):
        t = XX[:,:,:1]
        x = XX[:,:,1:2]
        u_0 = XX[:,:,2:2+b1]
        s_true = XX[:,:,-1:]
        
        s = self.makeNetwork(t,x,u_0)
        
        loss = s_true - s
        
        return torch.mean(loss**2)
    
    def loss_bc1(self, X):
   
        t = X[:,:,:1]
        x0 = X[:,:,1:2]
        x1 = X[:,:,2:3]
        u_0 = X[:,:,3:3+b1]
        
        s0 = self.makeNetwork(t,x0,u_0)
        s1 = self.makeNetwork(t,x1,u_0)
        
        loss = s0-s1

        return torch.mean(loss**2)
    
    def loss_bc2(self, X):
   
        t = X[:,:,:1]
        x0 = X[:,:,1:2].requires_grad_()
        x1 = X[:,:,2:3].requires_grad_()
        u_0 = X[:,:,3:3+b1]
        
        s0 = self.makeNetwork(t,x0,u_0)
        s1 = self.makeNetwork(t,x1,u_0)
        
        s0_x = torch.autograd.grad(s0, x0, grad_outputs=torch.ones_like(s0),  create_graph=True)[0]
        s1_x = torch.autograd.grad(s1, x1, grad_outputs=torch.ones_like(s1),  create_graph=True)[0]
        
        loss = s0_x-s1_x

        return torch.mean(loss**2)
    
    def train(self, num_epochs, lr, X_pde, X_ic, X_bc):
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.network.train()
        best_loss = float('inf'); loss_tracker = []
        
        def closure():
            self.optimizer.zero_grad()
            loss_pde = self.loss_pde(X_pde)
            loss_ic = self.loss_ic(X_ic)*8
            loss_bc1 = self.loss_bc1(X_bc)*10
            loss_bc2 = self.loss_bc2(X_bc)*5
            objective = loss_pde + loss_ic + loss_bc1 + loss_bc2
            objective.backward()
            return objective
        
        for epoch in range(num_epochs):
                 
            self.optimizer.step(closure)
            loss = closure()
            loss_tracker.append(loss.item())

            print('epoch {}/{}, loss = {:10.2e}'.format(epoch, num_epochs, loss))

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.network, 'burger_best_saved_mionet.pth')

            if epoch == int(0.8 * num_epochs): 
                self.optimizer = torch.optim.LBFGS(self.network.parameters(),
                                line_search_fn="strong_wolfe")
        
        epo = range(len(loss_tracker))
        fig, ax1 = plt.subplots(1,1,figsize=(6,4))
        ax1.set_yscale('log')
        ax1.plot(epo,loss_tracker,'b',label='Training loss')
        ax1.set_title('LOSS')
        ax1.legend()
        ax1.grid(visible=True)
          
    def predict(self, res, dT, U_0):
        
        T = np.ones([1,res,1])*dT
        
        Ksi = np.zeros((1,res,1))
        Ksi[0,:,0] = np.linspace(0,1,res)
        
        t = torch.tensor(T,dtype=torch.float32)
        ksi = torch.tensor(Ksi,dtype=torch.float32)
        u_0 =torch.tensor(U_0,dtype=torch.float32)

        y = self.makeNetwork(t,ksi,u_0)
        
        ## .detach().numpy()  needs to convert tensor in numpy
        return y.detach().numpy()[0,:,:1].T

def data_pde(N_pde, Ny):
    X_pde = np.zeros([N_pde, Ny, 2 + b1 + 1])
    soboleng_pde = torch.quasirandom.SobolEngine(dimension=2)
    coll_points_pde =  soboleng_pde.draw(Ny).detach().numpy()
    X_pde[:,:,:2] = coll_points_pde
    
    for idx in range(N_pde):
        scale_length = abs(np.random.normal(0,1))
        noise = np.random.normal(0, 1, (b1,))
        X_pde[idx,:,2:2+b1] = gaussian_filter(noise, sigma=scale_length)*scale_length # * scale_length needs to normalize the generated grf into the interval [0,1]

    return torch.tensor(X_pde, dtype=DTYPE, device= DEVICE) 

def data_ic(N_ic, Ny):
    X_ic = np.zeros([N_ic, Ny, 2 + b1 + 1])
    soboleng_ic = torch.quasirandom.SobolEngine(dimension=1)
    coll_points_ic =  soboleng_ic.draw(Ny).detach().numpy()
    sensor_points = np.linspace(0,1,b1) #Position of sensors
    X_ic[:,:,1:2] = coll_points_ic
    
    for idx in range(N_ic):
        scale_length = abs(np.random.normal(0,1))
        noise = np.random.normal(0, 1, (b1,))
        grf = gaussian_filter(noise, sigma=scale_length)*scale_length # * scale_length needs to normalize the generated grf into the interval [0,1]
        u_0 = grf
        u_x = np.interp(coll_points_ic[:,0], sensor_points, u_0)
        X_ic[idx,:,2:2+b1] = u_0
        X_ic[idx,:,-1] = u_x

    return torch.tensor(X_ic, dtype=DTYPE, device= DEVICE) 

def data_bc(N_bc, Ny):
    X_bc = np.zeros([N_bc, Ny, 3 + b1 + 1])
    soboleng_bc = torch.quasirandom.SobolEngine(dimension=1)
    coll_points_bc =  soboleng_bc.draw(Ny).detach().numpy()
    X_bc[:,:,:1] = coll_points_bc
    X_bc[:,:,0] = 1 # radial coordiante for enforcing bc1
    X_bc[:,:,2] = 1 # radial coordiante for enforcing bc2
    
    for idx in range(N_bc): 
        scale_length = abs(np.random.normal(0,1))
        noise = np.random.normal(0, 1, (b1,))
        X_bc[idx,:,2:2+b1] = gaussian_filter(noise, sigma=scale_length)*scale_length # * scale_length needs to normalize the generated grf into the interval [0,1]

    return torch.tensor(X_bc, dtype=DTYPE, device= DEVICE) 

DEVICE = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = device = torch.device('cpu')
DTYPE = torch.float32
b1 = 25 # size of branch net input 
viscosity = 0.1

## Preparing data: 
X_pde = data_pde(500, 2000)
X_ic = data_ic(100, 100)
X_bc = data_bc(10,20)

new_model = True; training = True

if new_model == True:
    net = DeePOnet(b1, 2, 50, 4)
    
else:
    net = torch.load('burger_best_saved_mionet.pth', weights_only=False)
    
net.to(DEVICE)
pinn = PINN(net)

if training: 
    num_epochs = 101; lr = 1e-4
    pinn.train(num_epochs, lr, X_pde,X_ic, X_bc)

#################### Predicted solution ####################
sensor_points = np.linspace(0,1,b1) #Position of sensors
noise = np.random.normal(0, 1, (b1,))
scale_length = 0.5
grf = gaussian_filter(noise, sigma=scale_length)*scale_length # * scale_length needs to normalize the generated grf into the interval [0,1]
u_0 = grf
time = np.linspace(0,1,101); Nx = 20
s = np.zeros((len(time),Nx))

k = 0
for t in time:
    s[k,:] = pinn.predict(Nx,t,u_0)[0,:]
    k+=1

#Plotting
fig, (ax,ax1,ax2) = plt.subplots(1,3,layout='constrained',figsize=(10,5))
cmap = plt.colormaps['bwr'] # you can chose different color maps
r = np.linspace(0,1,Nx)
im=ax.pcolormesh(r,time,s, cmap = cmap)
ax.set_title('Predicted')
ax.set_xlabel('x')
ax.set_ylabel('t')
fig.colorbar(im, ax=ax)

## Parameters
Nt = len(time)  # Number of time steps
x = np.linspace(0, 1, Nx)  # spatial grid
dx = x[1] - x[0]  # spatial step size
dt = time[1]  # time step size
rr = viscosity * dt / dx**2  # diffusion coefficient

u_0_ref = np.interp(x, sensor_points, u_0)
s_true = np.zeros((Nt, Nx))
s_true[0, :] = u_0_ref  # Set initial condition

# Numerical solution with finite differences
for n in range(1, Nt):
    for i in range(1, Nx-1):
        # Update rule:
        s_true[n, i] = s_true[n-1, i] - s_true[n-1, i] * dt / dx * (s_true[n-1, i] - s_true[n-1, i-1]) + rr * (s_true[n-1, i+1] - 2 * s_true[n-1, i] + s_true[n-1, i-1])

#Plotting
im=ax1.pcolormesh(x,time,s_true, cmap = cmap)
ax1.set_title('True')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
fig.colorbar(im, ax=ax1)

error = s_true - s 
im=ax2.pcolormesh(x,time,error, cmap = cmap)
ax2.set_title('Error')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
fig.colorbar(im, ax=ax2)

# Plot the results
plt.figure(figsize=(8, 6))
for n in range(0, Nt, Nt // 4):
    plt.plot(x, s[n, :],'r', label=f'Pred @ t={n*dt:.2f}')
    plt.plot(x, s_true[n, :],'*', label=f'Ref @ t={n*dt:.2f}')
    
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title("Solution of Burgers' Equation")
plt.legend()
plt.show()

def plot_idx(idx):
    plt.figure()
    plt.plot(sensor_points, u_0.T,'k--',label='IC')
    plt.plot(r,s[idx,:],'r',label='Predicted')
    plt.plot(x,s_true[idx,:],'*',label='Reference')
    plt.title('Imposed initial condition')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()

idx = 50
plot_idx(idx)
