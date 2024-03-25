## AONN_BFGS: Solving steady-state optimal control problems by L-BFGS in Hilbert space with physics-informed neural networks

The idea of applying gradient descent to solve optimal control problems mainly follows the paper (https://epubs.siam.org/doi/full/10.1137/22M154209X). In our work we try to improve its convergence by applying an L-BFGS method in Hilbert space. Details are included in AONN_BFGS/notes/note.pdf. 

The codes for linear unconstrained, linear constrained, semilinear unconstrained and steady state NS equation are included in folders linear, linear_ctd, semilinear and steady_NS, respectively. Dataset is generated in data.py. Files AONN and AONN_BFGS applied gradient descent and L-BFGS, respectively. 

## A modified natural gradient descent for neural networks via orthogonal greedy approximation

This work is motivated by the relavant paper Achieving High Accuracy with PINNs via Energy Natural Gradient Descent (https://proceedings.mlr.press/v202/muller23b.html). Optimizing on the Hessian manifold induced by the PINN loss function is found to yield very accurate PDE solutions, breaking the saturation phenomena of common PINN solutions. This work tries to address several issues in this method. The natural gradient descent is formulated as a continuous function approximation problem, and orthogonal greedy approximation is applied. Details can be found in notes Natural_gradient/notes_ngd/note.pdf. 

The performance is compared on least squares problem and PINN. For least squares problem, we compare Adam, Natural Gradient Descent and NGD with greedy approximation. The codes are included in Natural_gradient/NGD. For PINN, we compare Energy NGD with or without a greedy approximation. The codes are modified from the original work (https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23). All codes are implemented in JAX.

## Solving shape optimization problems with physics-informed neural networks by gradient descent method

This work applies the AONN approach to solve shape optimization problem, which is a special type of optimal control problems with the domain regarded as the control. This work follows the paper: AONN-2: An adjoint-oriented neural network method for PDE-constrained shape optimization (https://arxiv.org/abs/2309.08388). 

Codes for a linear elliptic example and an steady-state NS equation example are included in the folders linear and NS respectively. Dataset is generated in data.py, and solution is generated in AONN.py. 

## An implementation of model predictive control for stabilization and tracking of autonomous cars. 

This work follows the MPC toolbox in Matlab (https://ww2.mathworks.cn/help/mpc/ug/obstacle-avoidance-using-adaptive-model-predictive-control.html) and the course note from Bemporad (http://cse.lab.imtlucca.it/~bemporad/mpc_course.html). 

The codes finishes the stablization and tracking of an autonomous car. The class for matrix computation is included in controller.py, and the results are generated in the file test_stablizing.ipynb.

