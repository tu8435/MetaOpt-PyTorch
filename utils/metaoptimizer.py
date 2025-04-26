from typing import Dict
import torch
# Need baseline implementation for comparison... want to ensure deterministic result and prevent 
# _scaled_dot_product_efficient_attention_backward is not implemented error
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

from utils.utils import get_param_info
from utils.utils import unflatten_params
from utils.utils import make_step_fn


from matplotlib import pyplot as plt
from torch.func import functional_call
import time

import torch.nn as nn
from torch.optim import Optimizer
from collections import deque

#######################################
# The MetaOpt class
#######################################
class MetaOpt(Optimizer):
  """
  A meta-optimizer that:
  - subclasses torch.optim.Optimizer
  - uses an "inner" gpc_optim to update GPC parameters
  - produces final updates to the broader model as the combination
    of the base optimizer update and the meta-optimizer update
  """

  def __init__(
      self,
      model: nn.Module,
      H: int,
      HH: int,
      m_method: str,
      base_lr: callable, 
      weight_decay: float,
      freeze_gpc_params: bool,
      fake_the_dynamics: bool, 
      lr_gpc: float = 1e-3,
      device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"), # also configure for mps
      base_optimizer_cls: Optimizer = torch.optim.Adam,
      base_optimizer_kwargs: dict = {"lr": 1e-3, "betas": (0.9, 0.99)},
      gpc_optimizer_cls: Optimizer = torch.optim.SGD,
      gpc_optimizer_kwargs: dict = {"lr": 1e-3},
      max_norm: float = 1.0, # for gradient clipping the GPC params as they tend to be unstable
  ):
        """
        :param H: number of past gradients to store (This is the more important hyperparameter for practical consideration)
        :param HH: rollout length (How many steps into the future to play the rollout)
        :param m_method: 'scalar' | 'diagonal' | 'full' => controls the complexity of meta-control
        :param model: the neural net or module
        :param base_lr: function that returns LR for the base updates in the unrolled steps
        :param weight_decay: base weight decay for the unrolled steps
        :param freeze_gpc_params: whether to skip updating GPC params. Set this to False to learn optimizer, and True to deploy it
        :param fake_the_dynamics: whether to use the gradient buffer to time-evolve the system rather than taking bona fide train_steps during counterfactual rollout
        :param base_optimizer_cls: e.g. torch.optim.Adam
        :param base_optimizer_kwargs: e.g. dict(lr=1e-3, betas=(0.9,0.99))
        :param gpc_optimizer_cls: e.g. torch.optim.Adam
        :param gpc_optimizer_kwargs: e.g. dict(lr=1e-3, betas=(0.9,0.99))
        """
    
        # 1) Store basic params
        self.model = model # we'll store a reference to the original model but won't do in-place updates to its parameters
        self.device = device
        self.H = H
        self.HH = HH
        self.m_method = m_method
        self.base_lr = base_optimizer_kwargs["lr"]
        self.lr_gpc = lr_gpc
        self.weight_decay = weight_decay
        self.freeze_gpc_params = freeze_gpc_params
        self.fake_the_dynamics = fake_the_dynamics
        self.max_norm = max_norm
        self.t = 0
        self.meta_optimizer = None
        self.base_optimizer_cls = base_optimizer_cls
        self.base_optimizer_kwargs = base_optimizer_kwargs
        self.gpc_optimizer_cls = gpc_optimizer_cls
        self.gpc_optimizer_kwargs = gpc_optimizer_kwargs
        self.max_norm = max_norm # TODO: add support for gradient clipping of the meta optimizer update

        # choose & store the per-step function for the rollout
        # this is the functional version of the base optimizer
        self._functional_base_step = make_step_fn(base_optimizer_cls, base_optimizer_kwargs)
        self._functional_base_state: Dict = {}          # state container used by the step fn

        # 2) Flatten the model's parameters for shape info
        # It is ESSENTIAL that we only use trainable_parameters to build the meta-optimizers parameter and disturbance history to save on memory.
        # A primary limitation of Meta-opt is its large memory footprit as H and HH grow... consider using PEFT with Meta-opt to save on memory
        self.trainable_params = [p for p in model.parameters() if p.requires_grad] # --> SAVE MEMORY
        self.param_info = get_param_info(model)
        # filtered to requires_grad=True to obtain total trainable parameters
        self.param_info = [(name, shape, numel) for (name, shape, numel), p in zip(get_param_info(model), model.parameters()) if p.requires_grad]
        self.param_size = sum(info[2] for info in self.param_info) 


        # 3) Initialize GPC parameters and define container
        if m_method == "scalar":
            tensor_on_device = torch.zeros(H, device=self.device)
            self.gpc_params = nn.Parameter(tensor_on_device, requires_grad=True)
        elif m_method == "diagonal":
            tensor_on_device = torch.zeros(H, self.param_size, device=self.device)
            self.gpc_params = nn.Parameter(tensor_on_device, requires_grad=True)
        elif m_method == "full":
            tensor_on_device = torch.zeros(H, self.param_size, self.param_size, device=self.device)
            self.gpc_params = nn.Parameter(tensor_on_device, requires_grad=True)
        else:
            raise NotImplementedError(m_method)

        # create a container for gpc_params
        self.gpc_params_container = [self.gpc_params]

        # 4) Construct param_groups for the base optimizer and the meta-optimizer (let meta-opt look like a single optimizer to the caller)
        base_param_list = [p for p in model.parameters() if p.requires_grad]
        defaults = {}
        super().__init__([
            {"params": base_param_list, "lr": base_optimizer_kwargs["lr"], "meta": False},      # normal base param group, EXPLICITLY add lr
            {"params": self.gpc_params_container, "lr": gpc_optimizer_kwargs["lr"], "meta": True},  # GPC param, pass the container, not the parameter directly
        ], defaults)


        #5 ) Build the *internal* optimizer for updating gpc_params and base optimizer for base params
        self.gpc_optim  = gpc_optimizer_cls(self.gpc_params_container, **gpc_optimizer_kwargs)
        print(f"Using GPC optimizer: {gpc_optimizer_cls.__name__} with kwargs: {gpc_optimizer_kwargs}")
        self.base_optim = base_optimizer_cls([p for p in model.parameters() if p.requires_grad], **base_optimizer_kwargs)
        print(f"Using base optimizer: {base_optimizer_cls.__name__} with kwargs: {base_optimizer_kwargs}")


        # 6) Instantiate disturbances, param history, and data buffers
        self.disturbance_history = torch.zeros(self.H + self.HH, self.param_size, device=self.device)
        self.dist_ptr = 0 # keep index that increments

        self.param_history = torch.zeros(self.HH, self.param_size, device=self.device)
        self.param_ptr = 0

        self.data_buffer = deque(maxlen=self.H + self.HH)

  @torch.no_grad()
  def zero_grad(self, set_to_none: bool = False):
    """
    Overridden to zero out the GPC param's grad. Also call the gpc_optim's zero_grad.
    """
    self.base_optim.zero_grad(set_to_none=set_to_none)
    
    #TODO: REDUNDANT (it is the job of this optimizer to manually zero out the gpc_optim gradients)
    self.gpc_optim.zero_grad(set_to_none=set_to_none) 

  def compute_gpc_control(self, disturbance_history: torch.Tensor):
    """
    Computes the GPC control signal based on the current disturbance and parameter histories.
    """
    EINSUM_STRS = {"scalar": 'h,hn->n', "diagonal": 'hn,hn->n', "full": 'hmn,hn->m'}
    return torch.einsum(EINSUM_STRS[self.m_method], self.gpc_params, disturbance_history)

  def cost_fn_with_params(self, trainable_flat_params: torch.Tensor, inputs, batch_cost_fn):
    """
    Do a functional forward pass with 'trainable_flat_params' and the user's custom cost function for the data batch.
    """
    # It is the responsibility of the batch_cost_fn to do the forward pass after... 
    # self.param_info corresponds to the trainable parameters by default
    return batch_cost_fn(self.model, self.param_info, trainable_flat_params, inputs)


  # Be careful to check that x and y tensors are on the same device
  def functional_rollout(self):
    """
    If we have enough history, do a multi-step functional rollout from param_history[0],
    returning final_loss, final_params. Then we backprop that final_loss -> self.gpc_params.
    Update gpc_params using the inner gpc_optim.

    Simulate HH steps of rollout, starting from param_history[0].
    Each step uses base update plus GPC control.
    Then, compute the final loss, which depends on gpc_params.
    """

    # Insufficient History!!
    if self.t < (self.H + self.HH):
        return None, None

    # (1) Get param from HH steps ago
    # oldest param for the rollout is at ring idx
    rollout_start_idx = (self.param_ptr) % self.HH
    init_params = self.param_history[rollout_start_idx].clone().requires_grad_(True)
    current_params = init_params

    # We want the last HH data: from data_buffer
    # Because data_buffer is a deque, items are in order from oldest to newest
    # We'll just read the last HH entries
    # range(len(data_buffer) - HH, len(data_buffer))
    data_size = len(self.data_buffer)
    start = data_size - self.HH


    # (2) Simulate HH step rollout
    rollout_state = {} # keep local copies of the optimizer state that only live for the length of the rollout
    for step_i in range(self.HH):
      if self.fake_the_dynamics:
        # Use ring pointer for disturbance
        # e.g. the disturbance from H-1 + step_i
        ring_idx = (self.dist_ptr - self.H + step_i) % (self.H + self.HH)
        disturbance = self.disturbance_history[ring_idx]

      else:
        # Retrieve appropriate x_batch, y_batch, and cost_fn per batch to compute the loss
        inputs, batch_cost_fn = self.data_buffer[start + step_i]

        # Temporarily disable gradient checkpointing --> Temporary Workaround
        # TODO: Possibly integrate libraries like higher meant for higher order derivative calculations to prevent errors with gradient checkpointing
        # This shouldn't be of highest priority, given that gradient_checkpointing increases memory footprint, which Meta-Opt already struggles at
        if hasattr(self.model, "gradient_checkpointing_disable"):
          self.model.gradient_checkpointing_disable()

        # compute loss of model wrt current params using prior batch of data and gradients (GPC params will optimise around this)
        loss_ = self.cost_fn_with_params(trainable_flat_params=current_params, inputs=inputs, batch_cost_fn=batch_cost_fn)

        # Optionally, re-enable checkpointing after this computation if needed:
        if hasattr(self.model, "gradient_checkpointing_enable"):
          self.model.gradient_checkpointing_enable()


        # Now compute the gradient (this call now runs on a non-checkpointed forward graph)
        # partial derivatives of the loss wrt current_param model params
        grad_ = torch.autograd.grad(loss_, current_params, create_graph=True, allow_unused=False, materialize_grads=False)[0]
        disturbance = grad_

      # At this point loss_ has a path to controller self.gpc_params
      # We can detach the output of the step before making it the input of the next step
      # This prevents in-place mutations from F.opt() from corrupting earlier graph
      # Higher order gradients should still flow @ final_loss.backward() in the caller
      with torch.no_grad():
        # ------------------------- Simulate the base optimizer update -------------------------
        lr_base = self.base_lr(self.t + step_i) if callable(self.base_lr) else self.base_lr
        # Weight decay (as self.weight_decay -> 0, the params evolve naturally w/o shrinkag)
        decay_params = current_params * (1.0 - self.weight_decay)
        next_params, rollout_state = self._functional_base_step(decay_params, disturbance, rollout_state, lr_base)
        # --------------------------------------------------------------------------------------

        # ------------------------- Simulate the GPC optimizer update -------------------------
        # GPC control from last H disturbances
        # We'll gather the relevant H disturbances from the ring
        # For the i-th step, we want ring indexes [ring_ptr - H + step_i : ring_ptr + step_i)
        # We'll do a loop or slice
        control_indices = []
        # The ring pointer for "most recent" is self.dist_ptr - 1
        # but let's define a local pointer for "the step_i"
        # We'll skip complexities & do a small loop
        ptr = (self.dist_ptr - 1 + step_i) % (self.H + self.HH)
        for _ in range(self.H):
            control_indices.append(ptr % (self.H + self.HH))
            ptr -= 1
        control_indices.reverse()
        
        # gather
        past_grads = self.disturbance_history[control_indices]

      # compute GPC control and define next_params
      # VERY IMPORTANT: LET AUTOGRAD TRACK THE CONTROLLER CONTRIBUTIONS
      gpc_control = self.compute_gpc_control(past_grads)
      next_params += gpc_control
      # VERY IMPORTANT: DONT BREAK THE GRAPH HERE SO GRADS CAN FLOW
      current_params = next_params.clone().requires_grad_(True)
      # --------------------------------------------------------------------------------------

    # ------------------- Final loss on the last mini-batch of the rollout -------------------
    final_inputs, final_user_cost = self.data_buffer[start + self.HH - 1]
    final_loss = self.cost_fn_with_params(trainable_flat_params=current_params, inputs=final_inputs, batch_cost_fn=final_user_cost)
    # ----------------------------------------------------------------------------------------

    return final_loss, None # No need to return the params, but may be useful for logging/debugging
  

  def step(self, closure=None):
      """
      The standard "Optimizer.step()" call. Called once per iteration
      by HF Trainer or custom loop. We'll:
      (1) gather the newly computed grads from model.parameters() after loss.backward() was called in the outer function -> store in ring buffer
      (2) compute the final param update with [base optimizer + gpc control] -> apply to model.parameters()
      """
    
      if closure is None:
        raise RuntimeError("FunctionalMetaOpt requires a closure --> some function that returns a mini-batch/full set of datapoints, as well as cost function.")

      # 1) Gather grads and shift ring buffer
      new_grads = []

      for p in self.trainable_params:
        if p.grad is None:
          new_grads.append(torch.zeros_like(p.data.view(-1)))
        else:
          new_grads.append(p.grad.view(-1))
      new_grads_flat = torch.cat(new_grads).to(self.device)

      # shift in ring buffer
      self.disturbance_history[self.dist_ptr] = new_grads_flat
      self.dist_ptr = (self.dist_ptr + 1) % (self.H + self.HH)

      # 2) Gather model parameters and shift ring buffer
      with torch.no_grad():
        current_flat = nn.utils.parameters_to_vector(self.trainable_params)

      self.param_history[self.param_ptr] = current_flat
      self.param_ptr = (self.param_ptr + 1) % self.HH

      # 3) Gather closure and expand ring buffer
      inputs, cost_function = closure
      self.data_buffer.append((inputs, cost_function))

      # 4a) Normal base optimizer step for the base model... 
      # don't zero out base_optimizers' gradients here as gradient accumulation is handled by the HF trainer
      self.base_optim.step()

      # 4b) if we have enough H grads, apply the GPC control
      # gather last H from ring
      if self.t >= self.H:
          control_indices = []
          ptr = (self.dist_ptr - 1) % (self.H + self.HH)
          for _ in range(self.H):
              control_indices.append(ptr)
              ptr = (ptr - 1) % (self.H + self.HH)
          control_indices.reverse()
          past_grads = self.disturbance_history[control_indices]

          control_vec = self.compute_gpc_control(past_grads)
          # apply
          with torch.no_grad():
            
              # we have the current param as current_flat, but we must recalc it after the base_optim step:
              # or we can read the new param from the model again
              new_p = nn.utils.parameters_to_vector(self.trainable_params).to(self.device)
              updated_vec = new_p + control_vec
              nn.utils.vector_to_parameters(updated_vec, self.trainable_params)

      # 5) Perform the rollout to get gpc_loss from which to update gpc params
      if not self.freeze_gpc_params:
        gpc_loss, _ = self.functional_rollout()
        if gpc_loss is None:
          self.t += 1
          return # note enough history so don't perform an update step on GPC
        else:
          self.gpc_optim.zero_grad() # HF isn't handling the zero-ing out of the gradients wrt the gpc_params, only the model params, ensure we handle this here
          # NOTE: Forward pass that needs double backwards... but SDP Flash Attention is not implemented for double backwards
          # This is a known implementation gap, and thus any calls to trainer.train() needs to be wrapped in a context manage,r disabling SDP Flash Attention
          # See: https://github.com/pytorch/pytorch/issues/127523
          gpc_loss.backward()
          self.gpc_optim.step()
      self.t += 1
      return # gpc_loss.detach() loss should only be necessary for logging
