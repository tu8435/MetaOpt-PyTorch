# A Helpful test-file to ensure that Meta-Opt is working as expected if one is to make a change to the MetaOptimizer code
# In this multi-episodic setting, convex optimization setting, we expect the model using the meta-optimizer to converge to lower and lower loss
import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# ───────────────────────────── Models ──────────────────────────────
class SimpleModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.n_coeffs = 3  # Example for a 2nd-degree polynomial
      self.coeffs = nn.Parameter(torch.randn(self.n_coeffs, requires_grad=True))  # Random initialization

    def forward(self, x):
      # x: [batch_size, 1]
      powers = torch.arange(self.n_coeffs, device=x.device, dtype=x.dtype)
      # (x ** powers): [batch_size, n_coeffs]
      out = self.coeffs * (x ** powers)  # shape [batch_size, n_coeffs]
      # sum over dim=1, and keep dimension so it's [batch_size, 1]
      return out.sum(dim=1, keepdim=True)  # shape [batch_size, 1]

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),  # First hidden layer
            nn.ReLU(),
            nn.Linear(16, 16), # Second hidden layer
            nn.ReLU(),
            nn.Linear(16, 1)   # Output layer
        )

    def forward(self, x):
        return self.net(x)
# ─────────────────────────── Target functions ───────────────────────────
def convex_function(input_data: torch.Tensor):
  return (input_data**2) + 5

def non_convex_function(input_data: torch.Tensor):
  return (input_data**3) - (4 * input_data**2) + (2 * input_data) + 5

# ──────────────────────────── Cost fn  ─────────────────────────────
def example_cost_function(model, trainable_param_info, trainable_flat_params, inputs):
    from torch.func import functional_call            # torch ≥ 2.1
    from metaoptimizer import unflatten_params       

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs["device"], inputs["dtype"] = device, dtype

    trainable_param_dict = unflatten_params(trainable_flat_params, trainable_param_info)
    full_param_dict = dict(model.named_parameters())
    full_param_dict.update(trainable_param_dict)

    preds = functional_call(model, full_param_dict, inputs["input_ids"])
    return F.mse_loss(preds, inputs["labels"])

# ──────────────────────── Experiment loop  ─────────────────────────
def gaussian_noise_regression_compare_optimizers(
    *,
    convex: bool,
    net: nn.Module,
    base_optimizer_class: str,
    noise_intensity: float,
    max_norm: float,
    num_episodes: int,
    num_steps: int,
    fake_the_dynamics: bool = False,
):
    from metaoptimizer import MetaOpt  # <- adjust to your package layout

    optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam, "RMSprop": optim.RMSprop}
    if base_optimizer_class not in optim_dict:
        raise ValueError(f"Unknown optimizer {base_optimizer_class}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net.to(device)

    initial_state = copy.deepcopy(net.state_dict())
    x_train = torch.linspace(-2, 4, 100, device=device).view(-1, 1)
    y_train = torch.zeros(num_episodes, *x_train.shape, device=device)

    meta_opt = MetaOpt(
        model=net,
        H=5,
        HH=10,
        m_method="scalar",
        base_lr=lambda t: 1e-3,
        weight_decay=0.0,
        freeze_gpc_params=False,
        fake_the_dynamics=fake_the_dynamics,
        lr_gpc=1e-5,
        device=device,
        base_optimizer_cls=optim_dict[base_optimizer_class],
        base_optimizer_kwargs={"lr": 1e-3},
        max_norm=max_norm,
    )

    meta_losses, baseline_losses, times = [], {}, {"meta_opt": 0}

    t0 = time.time()
    for ep in range(num_episodes):
        y_fn = convex_function if convex else non_convex_function
        y_train[ep] = y_fn(x_train - torch.randn_like(x_train) * noise_intensity)

        net.load_state_dict(initial_state)
        for _ in range(num_steps):
            meta_opt.zero_grad()
            preds = net(x_train)
            loss = F.mse_loss(preds, y_train[ep])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm)

            inputs = {"input_ids": x_train, "labels": y_train[ep]}
            closure = lambda: (inputs, example_cost_function)
            meta_opt.step(closure=closure)      # ← call with *function*, not result
        meta_losses.append(loss.item())
        print(f"[Meta-Opt] ep {ep+1}/{num_episodes}  loss={loss.item():.4f}")

    times["meta_opt"] = time.time() - t0
    print("\nMeta-Opt finished → baselines …\n")

    for name, cls in optim_dict.items():
        t0 = time.time()
        opt = cls(net.parameters(), lr=1e-2)
        hist = []
        for ep in range(num_episodes):
            net.load_state_dict(initial_state)
            for _ in range(num_steps):
                opt.zero_grad()
                preds = net(x_train)
                loss = F.mse_loss(preds, y_train[ep])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm)
                opt.step()
            hist.append(loss.item())
            print(f"[{name}] ep {ep+1}/{num_episodes}  loss={loss.item():.4f}")
        baseline_losses[name] = hist
        times[name] = time.time() - t0

    # ──────────────────────── Plot + summary ───────────────────────
    plt.figure()
    plt.plot(meta_losses, label=f"Meta-Opt ({base_optimizer_class})")
    for k, v in baseline_losses.items():
        plt.plot(v, label=k)
    plt.xlabel("Episode")
    plt.ylabel("MSE loss")
    plt.title(f"{'Convex' if convex else 'Non-convex'} target")
    plt.legend()
    plt.show()

    print("\nFinal episode losses")
    print(f"Meta-Opt   {meta_losses[-1]:.6f}   time={times['meta_opt']:.1f}s")
    for k in optim_dict:
        print(f"{k:8} {baseline_losses[k][-1]:.6f}   time={times[k]:.1f}s")


# ────────────────────────── CLI interface ──────────────────────────
def str2bool(x: str) -> bool:
    return x.lower() in {"1", "true", "yes", "y", "t"}


def main():
    p = argparse.ArgumentParser(description="Meta-Opt Gaussian regression bench")
    p.add_argument("--convex", type=str2bool, default=True, help="fit convex target?")
    p.add_argument("--model", choices=["SimpleModel", "MyNet"], default="SimpleModel")
    p.add_argument("--base_optimizer_class", choices=["SGD", "Adam", "RMSprop"], default="Adam")
    p.add_argument("--noise_intensity", type=float, default=1.0)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--fake_the_dynamics", type=str2bool, default=False)
    args = p.parse_args()

    net_cls = SimpleModel if args.model == "SimpleModel" else MyNet
    gaussian_noise_regression_compare_optimizers(
        convex=args.convex,
        net=net_cls(),
        base_optimizer_class=args.base_optimizer_class,
        noise_intensity=args.noise_intensity,
        max_norm=args.max_norm,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        fake_the_dynamics=args.fake_the_dynamics,
    )


if __name__ == "__main__":
    main()
