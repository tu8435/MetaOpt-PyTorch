# MetaOpt-PyTorch
Take a look at my PyTorch Meta-Optimizer

# Lightweight_tests Usage
lightweight_tests is a tiny, self-contained benchmark suite meant to catch regressions any time you touch the MetaOpt optimizer or the surrounding helpers.
It solves a toy 1-D regression task under Gaussian noise and compares Meta-Opt (with your choice of base optimizer) against vanilla SGD/Adam/RMSprop.

**Why keep it around?**

**Fast** –finishes in seconds on CPU, < 1 s on GPU.

**Deterministic** – tiny problem size and fixed seeds give repeatable numbers.

**Actionable** – if Meta-Opt diverges or trains slower than a baseline, you’ll know immediately.

**Arugments** -
Flag | Default | Description
--convex / --no-convex | --convex | Fit the convex target x² + 5; toggle off to fit the non-convex cubic.
--model {SimpleModel, MyNet} | SimpleModel | Choose between a polynomial regressor and a 2-layer MLP.
--base_optimizer_class {SGD, Adam, RMSprop} | Adam | Optimizer Meta-Opt uses for GPC parameter updates.
--noise_intensity FLOAT | 1.0 | Standard deviation of Gaussian noise added per episode.
--max_norm FLOAT | 1.0 | Gradient clipping threshold. MetaOpt is sensitive to exploding gradients!
--num_episodes INT | 10 | Number of training episodes (outer loop).
--num_steps INT | 50 | Number of steps per episode (inner loop).
--fake_the_dynamics | False | Debug flag

Feel free to add more KWARGS and expand the comprehensiveness of these lightweight tests for your needs 😊!

**Example Usage -**
python lightweight_tests.py \
    --convex true \
    --noise_intensity 0.7 \
    --base_optimizer_class SGD



