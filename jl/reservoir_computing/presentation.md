# Reservoir Computing: Echo State Networks for Chaotic Prediction

Parth Bhargava · computational physics

Chaos puts a hard wall in front of prediction. Not a wall you can climb with a faster computer, but one set by the dynamics themselves: two trajectories that start a hair apart pull away exponentially, so past a certain horizon the future is simply unknowable from any finite measurement. This project builds an Echo State Network that gets as close to that wall as the physics allows, and it does it without backpropagation, training only a single linear layer by one matrix solve. Everything is in Julia. Each numbered section is a slide; the indented *Animation* line is the clip that plays on it, all rendered in `animations/output/`.

---

## Outline

1. The hook: why chaos caps prediction
2. Lyapunov exponents and the forecast ceiling
3. The Lorenz-63 system
4. The echo state property
5. Spectral radius and the edge of chaos
6. Building the reservoir
7. The readout: one matrix solve
8. Letting the network run on its own
9. The phase diagram
10. The interactive notebook
11. Where this goes next

---

## 1. The hook: why chaos caps prediction

Weather, turbulence, stirred fluids: they all share one obstruction, and it is not that the equations are expensive. It is that they are sensitive to initial conditions. Get the starting state slightly wrong and the error grows exponentially until the prediction is worthless, no matter how good the model is. Lorenz-63 is the smallest honest example of this, three coupled ODEs whose solutions live on a strange attractor.

A learned model has to recover the geometry of that attractor from noisy, finite data. Standard recurrent networks like LSTMs and GRUs can do it, but training them means backpropagation through time, which threads the gradient through every single step and blows up or vanishes once sequences run past a hundred steps or so. Echo State Networks dodge that entirely. They freeze the recurrent weights at random and train only a linear readout, which turns the whole learning problem into one closed-form matrix solve. The price is that the reservoir dynamics are fixed and random, and what makes that price worth paying is the *echo state property*: the guarantee that the reservoir's state remembers the recent input no matter how it was initialized.

> **Animation:** `s01_motivation.mp4`. Left: $x(t)$ for two Lorenz trajectories started $\varepsilon=0.001$ apart, tracking each other and then peeling away after about one Lyapunov time. Right: the separation $|\delta(t)|$ on a log scale, riding the exponential slope $\lambda_{\max}=0.906$, with the Lyapunov time $T_\lambda\approx1.1$ marked.

---

## 2. Lyapunov exponents and the forecast ceiling

Take two trajectories $\mathbf{x}(t)$ and $\mathbf{x}(t) + \boldsymbol{\delta}(t)$ that start an infinitesimal $\boldsymbol{\delta}(0) = \boldsymbol{\varepsilon}$ apart. The gap evolves under the linearized flow,

$$\dot{\boldsymbol{\delta}} = J(\mathbf{x}(t))\,\boldsymbol{\delta},$$

where $J$ is the Jacobian along the reference trajectory. The maximum Lyapunov exponent is the long-time average rate at which that gap grows:

$$\lambda_{\max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\boldsymbol{\delta}(t)\|}{\|\boldsymbol{\delta}(0)\|}.$$

For Lorenz-63 this converges to $\lambda_{\max} \approx 0.906\ \text{time}^{-1}$, the same value almost everywhere on the attractor. A positive $\lambda_{\max}$ is the definition of chaos: trajectories separate like $\varepsilon\,e^{\lambda_{\max} t}$, so after one Lyapunov time $T_\lambda = 1/\lambda_{\max} \approx 1.1$ the gap has grown by a factor of $e$.

That sets a ceiling you cannot buy your way past. Even with perfect dynamics, finite starting precision $\varepsilon_0$ means useful prediction ends when $\varepsilon_0\,e^{\lambda_{\max} t^*} \sim \sigma_{\text{attractor}}$, so $t^* = T_\lambda \ln(\sigma_{\text{attractor}} / \varepsilon_0)$. Operational weather models get roughly 7 Lyapunov times, about two weeks; the ESN here chases the same limit in miniature.

The full Lorenz spectrum is $(\lambda_1, \lambda_2, \lambda_3) \approx (0.906, 0, -14.6)$. The three add to $-13.67$, which is exactly the phase-space divergence $-(\sigma+1+\beta)$, so the attractor contracts volume. The zero exponent is the direction along the flow; the steep negative one yanks any transverse deviation back onto the attractor fast.

> **Animation:** `s02_lyapunov.mp4`. Left: the running Benettin estimate of $\lambda_{\max}$ converging to $0.906$ over 50 time units against a dashed reference. Right: the prediction ceiling $t^*=T_\lambda \ln(\sigma_{\text{att}}/\varepsilon_0)$ as bars for three starting precisions, $\varepsilon_0=10^{-2}, 10^{-4}, 10^{-8}$.

---

## 3. The Lorenz-63 system

The system itself (Lorenz 1963) is three lines:

$$\dot{x} = \sigma(y - x), \qquad \sigma = 10$$
$$\dot{y} = x(\rho - z) - y, \qquad \rho = 28$$
$$\dot{z} = xy - \beta z, \qquad \beta = \tfrac{8}{3}$$

These parameters sit it squarely in the chaotic regime. The fixed points at $(\pm\sqrt{\beta(\rho-1)}, \pm\sqrt{\beta(\rho-1)}, \rho-1)$ are unstable saddle-foci, and the attractor is the set of bounded orbits that wind around them without ever repeating. I integrate it with RK4 at a fixed step $\Delta t = 0.02$, so 50 time units of training data is 2500 steps, about 45 Lyapunov times of coverage.

> **Animation:** `s03_lorenz_attractor.mp4`. The butterfly building up over 30 time units in 3D, the two fixed points $C^+$ and $C^-$ marked, the camera slowly orbiting, and the equations labeled underneath.

---

## 4. The echo state property

An ESN is a discrete-time dynamical system driven by an input $\mathbf{u}(t) \in \mathbb{R}^d$. Its state $\mathbf{h}(t) \in \mathbb{R}^N$ steps forward as

$$\mathbf{h}(t+1) = \tanh\!\left(W_{\text{res}}\,\mathbf{h}(t) + W_{\text{in}}\,\mathbf{u}(t)\right).$$

The echo state property is the condition that makes this usable. It says that if you start two copies in different states $\mathbf{h}_1(0)$ and $\mathbf{h}_2(0)$ but feed them the same inputs, they forget the difference: $\|\mathbf{h}_1(t) - \mathbf{h}_2(t)\| \to 0$. When it holds, the reservoir state is a clean function of the input history alone, $\mathbf{h}(t) = \mathcal{E}[\mathbf{u}(t), \mathbf{u}(t-1), \ldots]$, and you can treat the reservoir as a nonlinear filter on the input rather than something that drags its own initial condition along forever.

The simple sufficient condition is that the spectral radius of $W_{\text{res}}$ is below one. To see why, look at the difference $\boldsymbol{\delta}(t) = \mathbf{h}_1(t) - \mathbf{h}_2(t)$ near zero input:

$$\boldsymbol{\delta}(t+1) = D(t)\,W_{\text{res}}\,\boldsymbol{\delta}(t),$$

where $D(t)$ is diagonal with $\tanh'$ entries, all at most 1 in magnitude. If $\rho(W_{\text{res}}) < 1$, repeatedly applying $D W_{\text{res}}$ contracts, and the gap dies geometrically. So $\rho < 1$ is both necessary (with no input) and sufficient for the linear part of the reservoir to be stable.

> **Animation:** `s04_echo_state.mp4`. Left: two reservoir traces $h_1^A(t)$ and $h_1^B(t)$ from different random starts, driven by the same Lorenz input, converging. Right: $\|h^A(t)-h^B(t)\|$ on a log scale washing out exponentially, the echo state property in one picture.

---

## 5. Spectral radius and the edge of chaos

The spectral radius does more than guarantee stability; it sets how long the reservoir remembers. For the undriven linearized reservoir $\mathbf{h}(t+1) \approx W_{\text{res}}\,\mathbf{h}(t)$, the state decays as $\|\mathbf{h}(t)\| \sim \rho^t$, so the memory depth is

$$T_{\text{mem}} \sim -1/\ln\rho.$$

At $\rho = 0.9$ that is about 9.5 steps; at $\rho = 0.99$, about 99. The whole design game is matching this memory depth to the correlation time of the signal you want to predict. Lorenz-63 stays correlated for roughly one Lyapunov time, around 55 steps at $\Delta t = 0.02$, so reservoirs with $\rho$ near 0.9 to 1.0 are tuned to it.

The edge of chaos is the boundary between ordered and chaotic dynamics, and for an ESN it sits right around $\rho \approx 1$:

- $\rho \ll 1$: strongly contracting, history is erased fast, and the readout has too little context to rebuild the attractor.
- $\rho \approx 1$: near-critical, state decays slowly, the last 50 to 100 inputs stay encoded, and the readout can approximate the Lorenz flow.
- $\rho > 1$: the undriven reservoir diverges; under bounded input it may stay bounded but the state gets dominated by reservoir transients rather than input, which kills the echo state property.

So the principle is to park $\rho$ in the marginally stable band where information is held, neither erased nor amplified.

> **Animation:** `s05_spectral_radius.mp4`. Left: undriven state norms $|h(t)|$ for $\rho=0.5, 0.8, 0.95, 1.2$, showing decay, sustain, and blow-up. Right: $T_{\text{mem}} = -1/\ln\rho$ against $\rho$, with a vertical line at the best sweep result $\rho=0.7$ and a horizontal reference at the 55-step Lorenz correlation time.

---

## 6. Building the reservoir

Three weight matrices, and only one of them is trained.

The input weights $W_{\text{in}} \in \mathbb{R}^{N \times d}$ are drawn from $\text{Uniform}(-s, +s)$ with scaling $s = 0.5$ and then frozen.

The reservoir weights $W_{\text{res}} \in \mathbb{R}^{N \times N}$ are a sparse Erdős–Rényi random graph: each entry is nonzero with probability $p_{\text{nz}}$, nonzero entries drawn from $\mathcal{N}(0,1)$. Then the whole matrix gets rescaled to hit the target spectral radius,

$$W_{\text{res}} \leftarrow W_{\text{res}} \cdot \frac{\rho_{\text{target}}}{\rho(W_{\text{res}})},$$

which sets the largest eigenvalue to exactly $\rho_{\text{target}}$ while leaving the sparsity pattern and relative weights alone. You only need the leading eigenvalue, so power iteration or `eigs` does it cheaply.

The output weights $W_{\text{out}} \in \mathbb{R}^{d \times (N+d)}$ are the only trained parameters, start at zero, and get solved by ridge regression in the next section.

The state vector that the readout sees concatenates reservoir and input, $\tilde{\mathbf{h}}(t) = [\mathbf{h}(t);\, \mathbf{u}(t)] \in \mathbb{R}^{N+d}$, so any linear part of the input can pass straight through without the reservoir having to encode it. The update runs with leak rate $\alpha = 1$, meaning no leaking,

$$\mathbf{h}(t+1) = \tanh\!\left(W_{\text{res}}\,\mathbf{h}(t) + W_{\text{in}}\,\mathbf{u}(t)\right).$$

The $\tanh$ is doing real work here. A linear reservoir would stay linear and could never represent the quadratic terms $xz$ and $xy$ in the Lorenz equations. Each bounded $\tanh$ unit is a small nonlinearity, and together they expand the input history into a high-dimensional nonlinear feature set. You want $N$ large enough for that expansion to be rich but small enough that the readout does not overfit, and the ridge term handles the balance.

> **Animation:** `s06_reservoir_init.mp4`. Left: a 20×20 sub-block of $W_{\text{res}}$ as a heatmap, showing the sparse wiring. Right: the eigenvalues of $W_{\text{res}}$ filling in across the complex plane, the unit circle dashed and the spectral-radius ring at $\rho=0.7$ marked.
>
> **Animation:** `s07_reservoir_dynamics.mp4`. Left: the Lorenz input $x(t), y(t), z(t)$ for the first 5 time units. Right: six reservoir traces $h_i(t)$ driven by it, showing the input history fanned out into a high-dimensional state.

---

## 7. The readout: one matrix solve

After a 100-step washout (thrown away to erase the dependence on $\mathbf{h}(0)=\mathbf{0}$), the remaining steps stack into a design matrix $X \in \mathbb{R}^{T_{\text{train}} \times (N+d)}$ with rows $\tilde{\mathbf{h}}(t)^{\top}$, and a target matrix $Y$ with rows $\mathbf{u}(t+1)^{\top}$, the one-step-ahead values. The output weights come from ridge regression:

$$W_{\text{out}} = \left(X^{\top}X + \lambda I\right)^{-1} X^{\top} Y, \qquad \lambda = 10^{-6}.$$

This is the Tikhonov-regularized least-squares solution, and it is closed-form because the loss $\mathcal{L}(W_{\text{out}}) = \|XW_{\text{out}}^{\top} - Y\|_F^2 + \lambda\|W_{\text{out}}\|_F^2$ is strictly convex and quadratic, so setting the gradient to zero gives a linear equation, $(X^{\top}X + \lambda I)W_{\text{out}}^{\top} = X^{\top}Y$.

Why not gradient descent? On a fixed reservoir it converges to this same answer, just slowly: $O(T \cdot N \cdot d)$ per epoch versus a single $O((N+d)^3)$ factorization. At $N=50$ the inversion is about $53^3 \approx 1.5\times10^5$ operations, which is nothing. The $\lambda I$ term also pushes the singular values of $X^{\top}X$ off zero, which keeps the solve stable even when reservoir nodes nearly synchronize, a common problem as $\rho$ approaches 1.

Why not backpropagation through time? If $W_{\text{res}}$ were trained too, the gradient would have to flow back through $T_{\text{train}}$ applications of $W_{\text{res}}$, piling up products of Jacobians with spectral radius near $\rho(W_{\text{res}})$. For $\rho<1$ those products shrink to nothing and for $\rho>1$ they explode. Freezing the reservoir sidesteps the entire problem.

> **Animation:** `s08_ridge_regression.mp4`. Left: training RMSE against $\log_{10}\lambda$ from $10^{-8}$ to $10^2$, the regularization tradeoff with the optimum at $\lambda=10^{-6}$ marked. Right: true $x(t)$ against the one-step-ahead prediction during training, drawing in.

---

## 8. Letting the network run on its own

At test time you warm the reservoir with one step of the true input $\mathbf{u}(0)$, then close the loop: feed the readout's own output back in as the next input, $\mathbf{u}(t+1) = \hat{\mathbf{u}}(t) = W_{\text{out}}\tilde{\mathbf{h}}(t)$. Now the ESN is generating its own trajectory on the attractor it learned, with no further help from the truth.

The prediction horizon is the first moment the normalized error crosses $\tau = 0.5$:

$$\text{PH} = \frac{t^*}{T_\lambda}, \qquad t^* = \min\left\{t : \frac{\|\hat{\mathbf{u}}(t) - \mathbf{u}(t)\|}{\|\mathbf{u}\|_{\text{rms}}} > \tau\right\}.$$

Dividing by the RMS amplitude of the true trajectory makes this attractor-invariant: a threshold of 0.5 means the error has reached half the typical signal size, which is a fair line for "no longer useful." Reporting $t^*$ in Lyapunov times strips out the system's own timescale and lets you compare across different chaotic systems.

The sweep that produces the results runs over a grid:

| Parameter | Values |
|-----------|--------|
| `sparsity` | 0.05, 0.10, 0.20, 0.40 |
| `spectral_radius` | 0.5, 0.7, 0.9, 0.95, 1.0, 1.1, 1.2, 1.5 |

Each of the $4 \times 8 = 32$ cells gets its own $N=50$ reservoir, trained on 2500 steps and tested over 500, scored by prediction horizon with RMSE as a secondary check. Every cell is independent, so the sweep parallelizes trivially.

> **Animation:** `s09_autonomous_pred.mp4`. Left: true $x(t)$ in blue against the ESN's autonomous $\hat{x}(t)$ in amber, tracking and then diverging. Right: the normalized error on a log scale against $t/T_\lambda$, the $\tau=0.5$ threshold dashed and the horizon of 1.96 Lyapunov times marked.

---

## 9. The phase diagram

The main result is a heatmap of prediction horizon over the grid of sparsity and spectral radius, and it has a clear structure.

The peak sits at the edge of chaos. The best horizons cluster around $\rho \approx 0.9$ to $1.0$, where the memory depth runs from about 45 steps up to effectively unbounded, matched to the Lorenz correlation time. Horizons of 3 to 5 Lyapunov times come routinely out of just 50 nodes.

Below that, for $\rho = 0.5$ to $0.7$, the reservoir is stable with room to spare but its memory is under 15 steps, too short for the readout to reconstruct the quadratic Lorenz terms, and horizons sag to 0 to 2 Lyapunov times. Above it, for $\rho = 1.2$ to $1.5$, the state grows even under bounded input, saturates the $\tanh$ units, collapses to near-binary switching, and the horizon falls to almost nothing.

Sparsity matters too. Sparse reservoirs ($p_{\text{nz}} = 0.05$ to $0.10$) beat dense ones at fixed $\rho$, because dense wiring correlates the reservoir states, drops the effective rank of $X$, and starves the ridge readout. Sparse random graphs spread activation broadly and have better spectral-gap behavior.

For scale: 50 nodes give 53 readout features against 2500 training steps, a ratio of about 47, comfortably overdetermined and safe from overfitting at $\lambda=10^{-6}$. At $N=500$ that ratio drops to about 4.9, and the regularization starts to matter much more.

In the best corner ($\rho \approx 0.95$, $p_{\text{nz}} = 0.05$) the ESN shadows the true trajectory for 3 to 5 Lyapunov times, then diverges onto the correct attractor lobe but out of phase. That shadow behavior, statistically right but phase-decorrelated, is exactly what a good learned chaotic model should look like: it has the geometry, and exponential divergence makes point-by-point long-range prediction impossible for any model at all.

> **Animation:** `s10_phase_diagram.mp4`. The prediction-horizon heatmap over the 4×8 grid, clipped to $[0,4]$ Lyapunov times, filling in column by column as $\rho$ grows, with the best cell annotated.
>
> **Animation:** `s11_trajectory_recon.mp4`. The true Lorenz trajectory in blue and the ESN's autonomous one in amber from the same start, agreeing for about 2 Lyapunov times then parting, both staying on the attractor while the camera orbits.

---

## 10. The interactive notebook

The Pluto notebook (`app.pluto.jl`) runs the whole pipeline behind sliders.

The reservoir panel has sliders for size $N$ (25 to 200), spectral radius $\rho$ (0.5 to 1.5), sparsity $p_\text{nz}$ (0.05 to 0.5), and ridge $\lambda$. Every change rebuilds and retrains the reservoir on the Lorenz data, and the state heatmap (nodes down, time across) updates so you can see dense versus sparse activation.

The prediction panel plots the autonomous ESN against the true trajectory for all three components, shows the normalized error trace and where it crosses threshold, and prints the prediction horizon in Lyapunov times. The 3D attractor plot puts the true (blue) and predicted (amber) orbits on the butterfly together until they split.

The phase-diagram panel runs the $4 \times 8$ sweep in the background, filling the horizon heatmap cell by cell, highlighting the optimal band and marking your current slider position with a white outline.

The experiment to do live: push $\rho$ to 1.5, above the ESP threshold, and watch the reservoir states diverge during washout and the horizon collapse to near zero. Then set $\rho=0.9$ and drop $N$ from 100 to 25, and watch the horizon fall from about 4 Lyapunov times to about 1.5, because the reservoir simply has less room to hold context.

> **Visual:** screenshot of `app.pluto.jl`, the reservoir heatmap on top, the prediction-versus-truth panel in the middle, and the phase-diagram heatmap at the bottom, mid-sweep with the $\rho\approx0.95$ band lit up.

---

## 11. Where this goes next

What is here: an Echo State Network that reaches the predictability limit of Lorenz-63 in miniature, trained gradient-free by a single matrix solve, with the whole edge-of-chaos story laid out as a phase diagram rather than asserted.

The headline is that one knob, the spectral radius, controls almost everything, because it sets the reservoir's memory depth, and the edge-of-chaos condition $\rho \approx 1$ is a genuine matching condition: the reservoir's timescale wants to equal the system's correlation time, which is the inverse Lyapunov exponent. The 5 to 7 Lyapunov-time ceiling is a fact about chaos, not the architecture, and no data-driven model beats it without exponentially better initial data. Reaching 3 to 5 of those times with 50 nodes and a linear solve is the reservoir-computing claim made concrete: nonlinear mixing plus a linear readout is enough to approximate a chaotic attractor.

A few honest caveats. Fifty nodes is enough to show the phenomenon but caps the horizon at 3 to 5 Lyapunov times; the scaling $\text{PH} \sim \log N / \lambda_{\max}$ says 8 times needs around 1200 nodes and weather-relevant horizons need $10^4$ to $10^5$. The sweep uses five realizations per cell and a fixed seed, so the cell-to-cell variation of about $\pm 1$ Lyapunov time is comparable to the differences it is resolving; the optimum is qualitatively right, not statistically sharp. For $\rho>1$ the washout never fully clears the initial state, so the diagram blends two distinct failure modes there. The readout is trained on one-step error and gives no guarantee the closed-loop state stays on the attractor. And Lorenz-63 is a three-variable toy with Kaplan-Yorke dimension about 2.06; high-dimensional systems like Lorenz-96 may need different reservoir sizes and possibly localized architectures, so this optimum need not transfer.

The natural extensions follow from those limits. Bigger reservoirs ($N = 500$ to $2000$) push the horizon up roughly as $\log N / \lambda_{\max}$ (Pathak et al. 2018). Lorenz-96, with 40 variables and spatially extended chaos, runs on the same architecture with only $N$ and $W_{\text{in}}$ growing. Deep ESNs stack reservoirs so each layer reads the one below and capture multi-timescale structure. A recursive-least-squares readout updates $W_{\text{out}}$ online via Sherman–Morrison–Woodbury, adapting to non-stationary attractors without storing the full design matrix. And next-generation reservoir computing (Gauthier et al. 2021) replaces the reservoir with a nonlinear delay embedding of the input, dropping $W_{\text{res}}$ entirely for comparable accuracy at almost no cost.
