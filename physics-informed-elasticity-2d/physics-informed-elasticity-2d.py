import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import trange

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Nf = 10000
N_b = 100

x_min, x_max = 0, 1
y_min, y_max = 0, 1

Q = 4
lmbd = 1
mu = 0.5

mse_loss = torch.nn.MSELoss()


def set_seed(seed=2026):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, activation='tanh'):
        super(NeuralNetwork, self).__init__()
        if hidden_size is None:
            hidden_size = [20, 20, 5]

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Softplus())

        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Softplus())

        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def auto_grad(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
    return auto_grad(auto_grad(u, x), x, order=order - 1)


def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


class PINNs():
    def __init__(
        self,
        layers, activation='tanh', device=device,
        initial_lr=0.001, sadap=False,
    ):
        self.device = device

        self.dnn = NeuralNetwork(
            input_size=layers[0], output_size=layers[-1], hidden_size=layers[1:-1],
            activation=activation
        ).to(device)
        self.dnn.apply(xavier_init)

        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=initial_lr,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.sadap = sadap
        self.optimizer_Adam = torch.optim.Adam(params=self.dnn.parameters(), lr=initial_lr)
        if self.sadap:
            self.scheduler = lr_scheduler.StepLR(self.optimizer_Adam, step_size=1000, gamma=0.9)

        self.iter = 0

    def closure(self):
        loss = self.lossf(model=self.dnn, td=None)

        self.iter += 1
        if self.iter % 100 == 0:
            print(f'Loss: {loss.item():.6f}')

        self.optimizer.zero_grad()
        loss.backward()
        return loss

    def train(self, epochs, lossf, refine_with_lbfgs=True):
        self.dnn.train()
        self.lossf = lossf

        with trange(epochs, dynamic_ncols=True) as td:
            for _ in td:
                loss = self.lossf(model=self.dnn, td=td)

                self.optimizer_Adam.zero_grad()
                loss.backward()
                self.optimizer_Adam.step()

                if self.sadap:
                    self.scheduler.step()

        if refine_with_lbfgs:
            print("LBFGS refinement...")
            self.optimizer.step(closure=self.closure)

    def predict1(self, x, y):
        self.dnn.eval().cpu()
        x = torch.tensor(x, requires_grad=True).float()
        y = torch.tensor(y, requires_grad=True).float()

        uxy = self.dnn(torch.cat([x, y], dim=1))

        epsilon_xx_pred = auto_grad(uxy[:, 0].reshape(-1, 1), x, 1).detach().numpy()
        epsilon_yy_pred = auto_grad(uxy[:, 1].reshape(-1, 1), y, 1).detach().numpy()
        epsilon_xy_pred = 0.5 * auto_grad(uxy[:, 1].reshape(-1, 1), x, 1) + \
            0.5 * auto_grad(uxy[:, 0].reshape(-1, 1), y, 1)

        epsilon_xy_pred = epsilon_xy_pred.detach().numpy()
        uxy = uxy.detach().numpy()
        self.dnn.to(self.device)

        return uxy, epsilon_xx_pred, epsilon_yy_pred, epsilon_xy_pred


# -fx
def bodyfx_torch(x, y):
    frc = - lmbd * (4 * torch.pi**2 * torch.cos(2 * torch.pi * x) * torch.sin(torch.pi * y) - Q * y**3 * torch.pi * torch.cos(torch.pi * x)) \
          - mu * (torch.pi**2 * torch.cos(2 * torch.pi * x) * torch.sin(torch.pi * y) - Q * y**3 * torch.pi * torch.cos(torch.pi * x)) \
          - 8 * mu * torch.pi**2 * torch.cos(2 * torch.pi * x) * torch.sin(torch.pi * y)
    return frc


# -fy
def bodyfy_torch(x, y):
    frc = lmbd * (3 * Q * y**2 * torch.sin(torch.pi * x) - 2 * torch.pi**2 * torch.cos(torch.pi * y) * torch.sin(2 * torch.pi * x)) \
          - mu * (2 * torch.pi**2 * torch.cos(torch.pi * y) * torch.sin(2 * torch.pi * x) + (Q * y**4 * torch.pi**2 * torch.sin(torch.pi * x)) / 4) \
          + 6 * Q * mu * y**2 * torch.sin(torch.pi * x)
    return frc


# u_x
def dispx_torch(x, y):
    return torch.cos(2 * torch.pi * x) * torch.sin(torch.pi * y)


# u_y
def dispy_torch(x, y):
    return torch.sin(torch.pi * x) * Q * y**4 / 4


# u_{x,x} == \epsilon_{xx}
def strain_x_x_torch(x, y):
    return -2 * torch.pi * torch.sin(2 * torch.pi * x) * torch.sin(torch.pi * y)


# u_{y, y} == \epsilon_{yy}
def strain_y_y_torch(x, y):
    return torch.sin(torch.pi * x) * Q * y**3


# \frac{1}{2} (u_{i,j} + u_{j,i}) == \epsilon_{xy}
def strain_xy_torch(x, y):
    return 0.5 * (torch.pi * torch.cos(2 * torch.pi * x) * torch.cos(torch.pi * y) + torch.pi * torch.cos(torch.pi * x) * Q * y**4 / 4)


def stress_xx_torch(x, y):
    return (lmbd + 2 * mu) * strain_x_x_torch(x, y) + lmbd * strain_y_y_torch(x, y)


def stress_yy_torch(x, y):
    return (lmbd + 2 * mu) * strain_y_y_torch(x, y) + lmbd * strain_x_x_torch(x, y)


# \sigma_{xy} == 2 \mu \epsilon_{ij}
def stress_xy_torch(x, y):
    return 2.0 * mu * strain_xy_torch(x, y)


def left_boundary(n=N_b, device=device):
    y = torch.rand(n, 1, device=device, requires_grad=True) * (y_max - y_min) + y_min
    x = torch.ones_like(y) * x_min
    return x, y


def right_boundary(n=N_b, device=device):
    y = torch.rand(n, 1, device=device, requires_grad=True) * (y_max - y_min) + y_min
    x = torch.ones_like(y) * x_max
    return x, y


def upper_boundary(n=N_b, device=device):
    x = torch.rand(n, 1, device=device, requires_grad=True) * (x_max - x_min) + x_min
    y = torch.ones_like(x) * y_max
    return x, y


def lower_boundary(n=N_b, device=device):
    x = torch.rand(n, 1, device=device, requires_grad=True) * (x_max - x_min) + x_min
    y = torch.ones_like(x) * y_min
    return x, y


def get_BC_Data(n=N_b, device=device):
    left_x, left_y = left_boundary(n=n, device=device)
    right_x, right_y = right_boundary(n=n, device=device)
    upper_x, upper_y = upper_boundary(n=n, device=device)
    lower_x, lower_y = lower_boundary(n=n, device=device)

    return torch.cat(
        (
            torch.cat((left_x, left_y), dim=1),
            torch.cat((right_x, right_y), dim=1),
            torch.cat((upper_x, upper_y), dim=1),
            torch.cat((lower_x, lower_y), dim=1),
        ),
        dim=0
    )


def boundary_loss_BC(model, boundary_fn=get_BC_Data, loss=mse_loss):
    boundaries_combined = boundary_fn()
    x, y = boundaries_combined[:, 0:1], boundaries_combined[:, 1:2]
    uxy = model(torch.cat([x, y], dim=1))

    u_x_true = dispx_torch(x, y)
    u_y_true = dispy_torch(x, y)
    sigma_xx_true = stress_xx_torch(x, y)
    sigma_yy_true = stress_yy_torch(x, y)
    sigma_xy_true = stress_xy_torch(x, y)

    return loss(uxy[:, 0].reshape(-1, 1), u_x_true) + \
        loss(uxy[:, 1].reshape(-1, 1), u_y_true) + \
        loss(uxy[:, 2].reshape(-1, 1), sigma_xx_true) + \
        loss(uxy[:, 3].reshape(-1, 1), sigma_yy_true) + \
        loss(uxy[:, 4].reshape(-1, 1), sigma_xy_true)


def interior(n=Nf, device=device):
    x = torch.rand(n, 1, device=device, requires_grad=True) * (x_max - x_min) + x_min
    y = torch.rand(n, 1, device=device, requires_grad=True) * (y_max - y_min) + y_min
    cond = torch.zeros_like(x)
    return x, y, cond


def pinn_pde_loss(model, interior_fn=interior, loss=mse_loss):
    x, y, cond = interior_fn()
    uxy = model(torch.cat([x, y], dim=1))

    u_x_pred = uxy[:, 0].reshape(-1, 1)
    u_y_pred = uxy[:, 1].reshape(-1, 1)
    sigma_xx_pred = uxy[:, 2].reshape(-1, 1)
    sigma_yy_pred = uxy[:, 3].reshape(-1, 1)
    sigma_xy_pred = uxy[:, 4].reshape(-1, 1)
    epsilon_xx_pred = auto_grad(u_x_pred, x, 1)
    epsilon_yy_pred = auto_grad(u_y_pred, y, 1)
    epsilon_xy_pred = 0.5 * auto_grad(u_y_pred, x, 1) + 0.5 * auto_grad(u_x_pred, y, 1)

    loss_all = loss(auto_grad(sigma_xx_pred, x, 1) + auto_grad(sigma_xy_pred, y, 1), bodyfx_torch(x, y)) + \
        loss(auto_grad(sigma_yy_pred, y, 1) + auto_grad(sigma_xy_pred, x, 1), bodyfy_torch(x, y)) + \
        loss((lmbd + 2 * mu) * epsilon_xx_pred + lmbd * epsilon_yy_pred - sigma_xx_pred, cond) + \
        loss((lmbd + 2 * mu) * epsilon_yy_pred + lmbd * epsilon_xx_pred - sigma_yy_pred, cond) + \
        loss(2 * mu * epsilon_xy_pred - sigma_xy_pred, cond)

    return loss_all


def get_pinn_loss(model, td=None, print_str=""):
    loss_F = pinn_pde_loss(model=model, interior_fn=interior)
    loss_B = boundary_loss_BC(model)
    loss_all = loss_F + loss_B

    if td is not None:
        td.set_description(
            f"pde: {loss_F.item():.5f}, bc: {loss_B.item():.5f}, "
            f"init: 0, total: {loss_all.item():.6f}" + print_str
        )

    return loss_all


# u_x
def dispx(x, y):
    return np.cos(2 * np.pi * x) * np.sin(np.pi * y)


# u_y
def dispy(x, y):
    return np.sin(np.pi * x) * Q * y**4 / 4


# u_{x,x} == \epsilon_{xx}
def strain_x_x(x, y):
    return -2 * np.pi * np.sin(2 * np.pi * x) * np.sin(np.pi * y)


# u_{y, y} == \epsilon_{yy}
def strain_y_y(x, y):
    return np.sin(np.pi * x) * Q * y**3


# \frac{1}{2} (u_{i,j} + u_{j,i}) == \epsilon_{xy}
def strain_xy(x, y):
    return 0.5 * (np.pi * np.cos(2 * np.pi * x) * np.cos(np.pi * y) + np.pi * np.cos(np.pi * x) * Q * y**4 / 4)


def stress_xx(x, y):
    return (lmbd + 2 * mu) * strain_x_x(x, y) + lmbd * strain_y_y(x, y)


def stress_yy(x, y):
    return (lmbd + 2 * mu) * strain_y_y(x, y) + lmbd * strain_x_x(x, y)


# \sigma_{xy} == 2 \mu \epsilon_{ij}
def stress_xy(x, y):
    return 2.0 * mu * strain_xy(x, y)


def cust_pcolor(AX, X, Y, C, title):
    im = AX.pcolor(X, Y, C, cmap="jet")
    AX.axis("equal")
    AX.axis("off")
    AX.set_title(title, fontsize=14)
    plt.colorbar(im, ax=AX)


def evaluate_and_plot(model, num_samples=100):
    x = np.linspace(x_min, x_max, num_samples)
    y = np.linspace(y_min, y_max, num_samples)
    X, Y = np.meshgrid(x, y)

    u_x_true = dispx(X, Y)
    u_y_true = dispy(X, Y)

    epsilon_xx = strain_x_x(X, Y)
    epsilon_yy = strain_y_y(X, Y)
    epsilon_xy = strain_xy(X, Y)

    sigma_xx = stress_xx(X, Y)
    sigma_yy = stress_yy(X, Y)
    sigma_xy = stress_xy(X, Y)

    u_pred, epsilon_xx_pred, epsilon_yy_pred, epsilon_xy_pred = model.predict1(
        X.flatten()[:, None], Y.flatten()[:, None]
    )

    for name, true, pred_idx in [
        ('Ux', u_x_true, 0), ('Uy', u_y_true, 1),
        ('Sxx', sigma_xx, 2), ('Syy', sigma_yy, 3), ('Sxy', sigma_xy, 4),
    ]:
        error = np.linalg.norm(
            true.reshape(-1, 1) - u_pred[:, pred_idx].reshape(-1, 1), ord=2
        ) / np.linalg.norm(true.reshape(-1, 1), ord=2)
        print(f'L2 Error {name}: {error:e}')

    fig, ax = plt.subplots(3, 2, figsize=(4, 6), dpi=300)
    cust_pcolor(ax[0, 0], X, Y, u_x_true, "Ux*")
    cust_pcolor(ax[0, 1], X, Y, u_y_true, "Uy*")
    cust_pcolor(ax[1, 0], X, Y, u_pred[:, 0].reshape(num_samples, num_samples), "Ux")
    cust_pcolor(ax[1, 1], X, Y, u_pred[:, 1].reshape(num_samples, num_samples), "Uy")
    cust_pcolor(ax[2, 0], X, Y, np.abs(u_pred[:, 0].reshape(num_samples, num_samples) - u_x_true), "abs error Ux")
    cust_pcolor(ax[2, 1], X, Y, np.abs(u_pred[:, 1].reshape(num_samples, num_samples) - u_y_true), "abs error Uy")
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("Displacement.png")

    fig, ax = plt.subplots(3, 3, figsize=(11, 11), dpi=300)
    cust_pcolor(ax[0, 0], X, Y, sigma_xx, "Sxx*")
    cust_pcolor(ax[0, 1], X, Y, sigma_yy, "Syy*")
    cust_pcolor(ax[0, 2], X, Y, sigma_xy, "Sxy*")
    cust_pcolor(ax[1, 0], X, Y, u_pred[:, 2].reshape(num_samples, num_samples), "Sxx")
    cust_pcolor(ax[1, 1], X, Y, u_pred[:, 3].reshape(num_samples, num_samples), "Syy")
    cust_pcolor(ax[1, 2], X, Y, u_pred[:, 4].reshape(num_samples, num_samples), "Sxy")
    cust_pcolor(ax[2, 0], X, Y, np.abs(u_pred[:, 2].reshape(num_samples, num_samples) - sigma_xx), "abs error Sxx")
    cust_pcolor(ax[2, 1], X, Y, np.abs(u_pred[:, 3].reshape(num_samples, num_samples) - sigma_yy), "abs error Syy")
    cust_pcolor(ax[2, 2], X, Y, np.abs(u_pred[:, 4].reshape(num_samples, num_samples) - sigma_xy), "abs error Sxy")
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("Stress.png")

    fig, ax = plt.subplots(3, 3, figsize=(11, 11), dpi=300)
    cust_pcolor(ax[0, 0], X, Y, epsilon_xx, "Exx*")
    cust_pcolor(ax[0, 1], X, Y, epsilon_yy, "Eyy*")
    cust_pcolor(ax[0, 2], X, Y, epsilon_xy, "Exy*")
    cust_pcolor(ax[1, 0], X, Y, epsilon_xx_pred.reshape(num_samples, num_samples), "Exx")
    cust_pcolor(ax[1, 1], X, Y, epsilon_yy_pred.reshape(num_samples, num_samples), "Eyy")
    cust_pcolor(ax[1, 2], X, Y, epsilon_xy_pred.reshape(num_samples, num_samples), "Exy")
    cust_pcolor(ax[2, 0], X, Y, np.abs(epsilon_xx_pred.reshape(num_samples, num_samples) - epsilon_xx), "abs error Exx")
    cust_pcolor(ax[2, 1], X, Y, np.abs(epsilon_yy_pred.reshape(num_samples, num_samples) - epsilon_yy), "abs error Eyy")
    cust_pcolor(ax[2, 2], X, Y, np.abs(epsilon_xy_pred.reshape(num_samples, num_samples) - epsilon_xy), "abs error Exy")
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("Strain.png")


def main():
    set_seed(2023)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Device:", device)

    layers = [2, 60, 60, 60, 60, 5]
    model = PINNs(
        layers=layers, activation='tanh', device=device,
        sadap=True, initial_lr=0.001,
    )

    model.train(epochs=20000, lossf=get_pinn_loss, refine_with_lbfgs=True)
    evaluate_and_plot(model)


if __name__ == '__main__':
    main()
