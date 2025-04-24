# MetaOpt-PyTorch
Take a look at my PyTorch Meta-Optimizer

# Lightweight_tests Usage
lightweight_tests is a tiny, self-contained benchmark suite meant to catch regressions any time you touch the MetaOpt optimizer or the surrounding helpers.
It solves a toy 1-D regression task under Gaussian noise and compares Meta-Opt (with your choice of base optimizer) against vanilla SGD/Adam/RMSprop.

**Why keep it around?**
**Fast – **finishes in seconds on CPU, < 1 s on GPU.

**Deterministic – **tiny problem size and fixed seeds give repeatable numbers.

**Actionable –** if Meta-Opt diverges or trains slower than a baseline, you’ll know immediately.

**Arugments -**
**flag**	                                 **default**	        **description**
--convex / --no-convex	                    --convex	            Fit the convex target x²+5; turn off to fit the cubic.
--model {SimpleModel,MyNet}	                SimpleModel	          Choose a tiny poly-regressor or a 2-layer MLP.
--base_optimizer_class {SGD,Adam,RMSprop}	  Adam	                Which torch optimizer Meta-Opt uses for its GPC parameters.
--noise_intensity FLOAT	                    1.0	                  Std-dev of Gaussian noise injected each episode.
--max_norm FLOAT	                          1.0	                  Gradient-clipping threshold. MetaOpt is very susceptible to gradient explosion!
--num_episodes INT	                        10	                  How many mini-tasks to run in the outer loop.
--num_steps INT	                            50	                  Inner-loop SGD steps per episode.
--fake_the_dynamics	                        False	                Debug switch that disables Meta-Opt’s learned dynamics (should only be True for unit tests).

Feel free to add more KWARGS and expand the comprehensiveness of these lightweight tests for your needs 😊!

**Example Usage -**
python lightweight_tests.py \
    --convex true \
    --noise_intensity 0.7 \
    --base_optimizer_class SGD



