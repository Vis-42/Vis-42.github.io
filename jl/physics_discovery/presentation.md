# Physics Discovery: SINDy, Conservation Laws, and Symbolic Regression

Parth Bhargava · computational physics

Newton got $F=ma$ by staring at planets for years. This project asks whether an algorithm can do the same thing from a table of numbers: hand it a trajectory, no physics attached, and have it hand back the law. Three different methods take a run at that question, each from its own angle, and the interesting part is where they agree and where they break. Everything below is built from scratch in Julia. Each numbered section is a slide; the indented *Animation* line is the clip that plays on it, all rendered in `animations/output/`.

---

## Outline

1. The hook: physics, run backwards
2. SINDy: guess a menu, then cross things off
3. Building the library, and the one step that hurts
4. Checking the answer by replaying it
5. Conserved quantities as an eigenvalue problem
6. The same trick on three real systems
7. Symbolic regression: evolving equations
8. The Feynman benchmark
9. Three methods, three different questions
10. Where each one falls apart
11. The interactive notebook
12. Where this goes next

---

## 1. The hook: physics, run backwards

The usual direction in physics is forward. You write down a law, you turn the crank, you predict what happens. This project runs the other way. You start with what happened, a recorded trajectory $\{x(t_1), x(t_2), \ldots, x(t_n)\}$, and you try to recover the law that produced it, ideally as a short equation you could write on a board.

Three methods get a turn, and they answer genuinely different questions:

- **SINDy** (Brunton et al. 2016) writes the equation of motion. Sparse regression picks a handful of terms out of a big menu of candidates.
- **Kernel conservation-law discovery** writes down what stays constant. It hunts for a function that does not drift along the trajectory.
- **Symbolic regression** evolves a formula directly, breeding expression trees until one fits.

Point all three at the same data and you get three different kinds of answer about the same physics. That contrast is the whole talk.

> **Animation:** `s01_motivation.mp4`. Three panels left to right: a Van der Pol trajectory, the equation SINDy pulls out of it, and the trajectory you get back by integrating that equation forward. The recovered curve lands on the original attractor, which is the point. The law was hiding in the data.

---

## 2. SINDy: guess a menu, then cross things off

SINDy starts from a bet: physical laws are short. Out of all the terms you could imagine appearing in $\dot{x}$, only a few actually do. So write the dynamics as a big linear combination and then force most of the coefficients to zero.

Stack a dictionary of candidate functions $\Theta(x)$ and look for a sparse coefficient matrix $\xi$:

$$\dot{X} = \Theta(X)\,\xi$$

The columns of $\Theta \in \mathbb{R}^{n \times p}$ are the candidates you are willing to entertain: $1, x_1, x_2, x_1^2, x_1 x_2, \sin x_1$, and so on. The matrix $\xi \in \mathbb{R}^{p \times d}$ says which ones survive and with what weight.

Sequentially Thresholded Least Squares (STLSQ) is the part that does the crossing-off:

1. Solve the full least-squares problem, $\xi^{(0)} = \Theta^+ \dot{X}$.
2. Zero out any coefficient smaller than $\lambda$.
3. Re-solve on the columns that are left.
4. Repeat until the surviving set stops changing.

It converges in a few passes. Van der Pol, $\dot{x}_2 = \mu(1 - x_1^2)x_2 - x_1$, uses exactly three terms out of a library of about twenty, and STLSQ finds those three.

The one knob that matters is $\lambda$. Set it too high and you zero out real terms and underfit. Set it too low and spurious terms ride along. The polynomial degree sets the menu size, $p = \binom{d+n}{n} - 1$, and you have to add trig by hand if the system is periodically driven, like the Duffing oscillator or the driven pendulum.

---

## 3. Building the library, and the one step that hurts

For a state of dimension $n$ and polynomial degree $d$, the count of monomials is

$$p_{\text{poly}} = \binom{d+n}{n} - 1.$$

At $n=2$, $d=3$ that is nine monomials plus a constant, ten terms, and adding the four trig terms takes you to fourteen. Small menus, which is exactly what makes the sparse fit well posed.

The painful step is getting $\dot{x}$ in the first place. You rarely measure it; you measure $x$ and difference it. A centered finite difference is $O(\Delta t^2)$ accurate,

$$\dot{x}_t \approx \frac{x_{t+1} - x_{t-1}}{2\Delta t},$$

but it amplifies noise by $1/\Delta t$. Double the measurement noise $\sigma$ and the noise on the derivative grows like $\sigma/\Delta t$. This single step sets the ceiling on everything SINDy can do.

Van der Pol at $\mu = 1$ shows it working cleanly:

| State | True equation | Identified |
|-------|--------------|------------|
| $\dot{x}_1$ | $x_2$ | $\xi_3 = 1.0$ on $x_2$ |
| $\dot{x}_2$ | $x_2 - x_1^2 x_2 - x_1$ | $\xi_3=0.97$, $\xi_7=-0.98$, $\xi_5=-1.01$ |

Recovery is exact at $\sigma=0$ and stays trustworthy up to about $\sigma \approx 0.05$, beyond which spurious terms start creeping in. The driven pendulum, $\dot{x}_2 = -b\,x_2 - \sin(x_1) + A\cos(\omega t)$, makes the menu choice concrete: a polynomial library of degree five or more fits $\sin(x_1)$ near the origin but falls apart past $x_1 > \pi/2$. Give it the trig term and it recovers the law exactly.

> **Animation:** `s02_sindy_library.mp4`. The $\Theta$ matrix as a heatmap, each column a candidate term colored by its coefficient. As the threshold $\lambda$ sweeps up, columns go gray one by one and the active set thins down to the three terms that are really there.

---

## 4. Checking the answer by replaying it

Recovering coefficients is not the same as recovering the physics. The honest test is to take the discovered model and integrate it forward from the true starting point with RK4:

$$\hat{x}(t_{k+1}) = \text{RK4}(\hat{f}_\xi,\, \hat{x}(t_k),\, \Delta t), \qquad \hat{f}_\xi(s) = \Theta(s)\,\xi.$$

Then ask whether the replayed trajectory tracks the original. Three things to look at: the in-sample RMSE between $\dot{X}_\text{true}$ and $\Theta\hat{\xi}$, how long the replayed phase portrait stays glued to the real one before they part, and whether the limit cycle comes out the right shape, within about 5% in enclosed area.

The noise story is graded, not all-or-nothing:

- $\sigma = 0.0$: perfect, RMSE below $0.01$.
- $\sigma = 0.02$: coefficients jitter a little, replay still valid over the full time span.
- $\sigma = 0.05$: a spurious term may show up, removable by nudging $\lambda$ higher.
- $\sigma > 0.1$: the derivative estimate is too corrupted and the whole thing breaks.

The Duffing oscillator carries a warning. It is non-autonomous, so you either augment the state to $[x_1, x_2, t]$ or hand it a trig library at the forcing frequency $\omega$. Skip that and SINDy fits an autonomous Duffing, which has the wrong topology and looks fine until you trust it.

> **Animation:** `s03_sindy_reconstruction.mp4`. Split screen in phase space: true trajectory in blue, RK4 replay in green, sharing a time cursor. An RMSE counter ticks up as the frames advance, so you watch agreement hold or fail in real time.

---

## 5. Conserved quantities as an eigenvalue problem

The second method asks a different question. Forget the equation of motion; find what stays constant. A conservative system has some function $H(x)$ with $dH/dt = 0$ along every trajectory, and energy is the obvious example.

The kernel approach (Champion et al.) looks for that function in a space of smooth functions built from radial basis functions, and picks the one that drifts least in time. Lay down anchor points $z_1, \ldots, z_m$ subsampled from the data and build the feature map

$$\Phi_{ij} = \exp\!\left(-\gamma\,\|x_i - z_j\|^2\right).$$

Now find the coefficients $c$ that make $\Phi c$ as flat in time as possible relative to its size:

$$\min_c \frac{\|\dot{\Phi}c\|^2}{\|\Phi c\|^2} \;\Rightarrow\; A\,c = \lambda\,B\,c,$$

with $A = (\Delta\Phi/\Delta t)^\top(\Delta\Phi/\Delta t) + \alpha I$ and $B = \Phi^\top\Phi + \varepsilon I$. This is a generalized eigenvalue problem, and the eigenvector with the smallest eigenvalue is the most conserved function the data will support. The output is $H_\text{kernel}(x) = \Phi(x)\,c$, which is a smooth surface over phase space rather than a formula. To read it as a formula you project it onto a polynomial basis afterward.

> **Animation:** `s05_kernel_invariant.mp4`. A four-panel reveal: $H_\text{kernel}$ plotted against the true $H$ for pendulum, Kepler, and Duffing as the points accumulate, plus the eigenvalue sweep $\rho(\gamma)$ dipping to a minimum at the right bandwidth, and the time trace $H(t)$ sitting flat.

---

## 6. The same trick on three real systems

Run it on three conservative systems and it locks onto the energy in each:

| System | True Hamiltonian | Kernel discovery |
|--------|-----------------|-----------------|
| Pendulum | $\tfrac{1}{2}\dot{\theta}^2 - g\cos\theta$ | Correlation $\rho > 0.99$ |
| Kepler orbit | $\tfrac{1}{2}v^2 - 1/r$ | Correlation $\rho > 0.98$ |
| Duffing | $\tfrac{1}{2}\dot{x}^2 + \tfrac{1}{4}x^4 - \tfrac{1}{2}x^2$ | Correlation $\rho > 0.97$ |

Kepler is the nice one. Its state is four-dimensional ($x, y, v_x, v_y$) and its energy is

$$H = \frac{1}{2}(v_x^2 + v_y^2) - \frac{1}{\sqrt{x^2+y^2}}.$$

It also conserves angular momentum, $L = xv_y - yv_x$, and the method finds that too, as the second eigenvector of the same eigenproblem. One trajectory, two invariants, no extra work.

The bandwidth $\gamma$ is the knob. Too small and the function comes out too smooth and misses the curvature near the energy extrema. Too large and it interpolates the noise and generalizes badly. The median heuristic, $\gamma \approx 1/\text{median}(\|x_i - x_j\|^2)$, lands in the right place almost every time.

> **Animation:** `s04_conservation_phase.mp4`. Three rows, one per system. The left column animates the phase portrait with a trailing tail; the right column shows $H(t)$ holding flat, which is conservation made visible.

---

## 7. Symbolic regression: evolving equations

The third method does not assume a menu at all. It writes mathematical expressions as binary trees, leaves being constants or variables and interior nodes being operations ($+, -, \times, \div, \sqrt{}, \sin, \cos$), and then breeds them. The trees that fit best get to reproduce.

Fitness rewards accuracy and punishes size, so the search does not drift toward huge tangled expressions:

$$\mathcal{F}(T) = \text{NMSE}(T, \mathbf{y}) + c_p \cdot |T|,$$

where $|T|$ is the node count. Drop the parsimony term and the trees bloat, growing subtrees that cancel out but pad the formula. The genetic operators are the standard three: crossover swaps a random subtree between two parents, mutation rewrites a single node, and the top 5% survive untouched as elites. Selection is by tournament, sampling two trees and keeping the fitter, which is gentle enough to keep the population diverse and stop it collapsing onto one answer too early.

> **Animation:** `s06_symbolic_gp.mp4`. Left panel: the fitness curve climbing generation by generation. Right panel: predicted against true values for the best tree, appearing about 70% of the way through, with the discovered expression string showing up at 75% and sharpening toward the final form.

---

## 8. The Feynman benchmark

To measure success you need cases where the right answer is known, so the test set is five equations from the Feynman Symbolic Regression Benchmark:

| Equation | Formula | Variables | $R^2$ at gen 50 |
|----------|---------|-----------|-------------|
| Kinetic energy | $E = \tfrac{1}{2}mv^2$ | $m, v$ | ~0.998 |
| Pendulum period | $T = 2\pi\sqrt{L/g}$ | $L$ | ~0.995 |
| Ohm's law | $V = IR$ | $I, R$ | ~0.999 |
| Hooke's law | $F = kx$ | $k, x$ | ~0.999 |
| Gravitational PE | $U = mgh$ | $m, g, h$ | ~0.997 |

These five are deliberately easy: one to three variables each, a known closed form to score against, and a spread across the basic operation types, products and ratios and a square root. With a population of 100, the simple ones land on the exact closed form in 20 to 50 generations. The pendulum period is the slow one, because the $\sqrt{}$ has to be in the operator set for the search to even reach it.

> **Animation:** `s07_gp_convergence.mp4`. Left: an $R^2$ bar chart filling toward the final values for all five equations. Right: the fitness curves over generations, with Ohm and Hooke converging well ahead of the square-root case.

---

## 9. Three methods, three different questions

Point all three at Van der Pol, the pendulum, and Kepler together and the division of labor is clear:

| Criterion | SINDy | Kernel invariants | Symbolic GP |
|-----------|-------|------------------|-------------|
| Output | ODE coefficients | Conserved function | Closed-form expression |
| Noise robustness | Moderate ($\sigma < 0.05$) | High (regularized) | Low (needs clean data) |
| Interpretability | Explicit equation | Implicit kernel | Symbolic formula |
| Scalability | With library size | With $m^2$ | Exponential in depth |
| Extrapolation | By integration | Only on attractor | Algebraic |
| Prior knowledge | Basis choice | None | Operator set |

They are not competitors so much as different jobs. SINDy hands you the equation of motion, so you can simulate forward from any new starting point. The kernel method hands you the conserved quantity, so you get the symmetry structure without ever integrating. GP hands you a portable formula, the kind of thing that goes in a paper or straight into code. A natural pipeline chains them: SINDy finds the dynamics, the kernel method finds the symmetries, and GP simplifies the kernel output into something symbolic.

> **Animation:** `s08_comparison.mp4`. Three panels on one Kepler dataset, building together: SINDy's coefficient bars on the left, the kernel energy surface $H(x_1, x_2)$ in the center, and the best GP expression tree on the right.

---

## 10. Where each one falls apart

Every method here has a hard edge, and naming it is the honest part.

**SINDy lives or dies on the derivative.** Centered differences amplify noise by $1/\Delta t$, so at $\Delta t = 0.02$ and $\sigma = 0.02$ the noise on $\dot{x}$ is around $1.0$, comparable to the signal itself for slowly varying states. Smoothing helps but trades noise for bias. Empirically the ceiling for clean recovery is $\sigma \lesssim 0.05$ on these systems; past it, spurious terms appear that thresholding cannot remove without also killing the real ones.

**SINDy can only find what is on the menu.** STLSQ returns the sparsest fit inside $\Theta$. If the true dynamics need a function the library does not contain, an exponential, a Bessel function, some non-polynomial forcing, no amount of data will recover it. The driven pendulum needs its trig term; without it you get the best polynomial fit, which is simply the wrong law once you leave the training region.

**The kernel method gives a surface, not a sentence.** $H = \Phi c$ is a smooth function over phase space, interpretable as a picture but not as a formula. Projecting it onto polynomials recovers a symbolic approximation, but Kepler's $1/r$ needs a rational basis that a polynomial projection can never represent exactly.

**Symbolic GP chokes on dimension.** For Kepler's four variables, the space of depth-four trees over ten operators is roughly $(10\cdot4)^4 \approx 2.6\times10^6$ per leaf layout, far past exhaustive search. Convergence slows hard above three variables; the Feynman cases here are favorable, and the full four-variable Kepler Hamiltonian needs more than 500 generations to land reliably.

**None of them quantify uncertainty.** SINDy returns a point estimate, the kernel method one eigenvector, GP one tree. There is no posterior over possible laws. Bootstrap ensembles, re-running on resampled trajectories, would give empirical error bars, and that is the obvious thing to add next.

---

## 11. The interactive notebook

The Pluto notebook (`app.pluto.jl`) puts all three methods behind sliders, three panels you can poke at instead of watching fixed clips.

The SINDy panel lets you pick the system (Van der Pol, Duffing, or driven pendulum) and dial noise $\sigma$ from 0 to 0.15, threshold $\lambda$ from 0.01 to 0.5, and polynomial degree from 1 to 4. Every change reruns STLSQ to convergence and redraws the identified equation, the recovered-versus-true coefficient bars, and the blue/green phase-portrait overlay.

The conservation panel solves the kernel eigenproblem live, with sliders for bandwidth $\gamma$ and regularization $\alpha$. You see the discovered $H_\text{kernel}(x)$ as a surface, the $H(t)$ trace holding flat, and the correlation $\rho$ against the true Hamiltonian as a number.

The symbolic panel runs GP in front of you on a chosen Feynman equation, population from 20 to 200 and generations from 10 to 100, redrawing the fitness curve and the current best expression each generation.

The experiment worth doing live is in the SINDy panel. Set $\sigma = 0$, $d = 3$, Van der Pol, and watch the coefficient bars snap to $\xi = [1, -1, -1]$ in three iterations. Then push $\sigma$ to 0.08. A spurious term appears, an $x_1^3$ or $x_2^3$, and you cannot threshold it away without also zeroing the real terms. That is the noise ceiling, shown rather than described.

> **Visual:** screenshot of `app.pluto.jl`, all three panels: SINDy coefficient bars on top, the Kepler energy surface in the middle, the GP fitness curve and expression string at the bottom.

---

## 12. Where this goes next

What is here: three genuinely different ways to pull physics out of raw trajectories, each checked against a system whose answer is known, each fast enough to drive live, and each with its failure mode stated rather than hidden.

| Method | System | Key result | Quantitative |
|--------|--------|------------|-------------|
| SINDy | Van der Pol | Exact ODE recovery | $\xi$ error $< 3\%$ at $\sigma = 0.02$ |
| SINDy | Driven pendulum | Correct trig terms | needs $\sin/\cos$ in library |
| Kernel | Kepler | Energy conserved | $\rho > 0.98$ vs true $H$ |
| Kernel | Pendulum | $H = \tfrac{1}{2}\dot{\theta}^2 - g\cos\theta$ | $\rho > 0.99$ |
| GP | Kinetic energy | Exact: $E = \tfrac{1}{2}mv^2$ | $R^2 = 0.998$, 50 gen |
| GP | Hooke's law | Exact: $F = kx$ | $R^2 = 0.999$, 20 gen |

Underneath all three is the same wager, that physical laws are simple: sparse for SINDy, low-drift for the kernel method, shallow trees for GP. That is not luck. It is the symmetry structure of classical mechanics showing up as compressibility, and Nature's apparent taste for short descriptions.

The extensions build straight on these engines. PDE-FIND (Schaeffer 2017) adds spatial derivatives to the SINDy library and recovers PDEs, demonstrated on Kuramoto-Sivashinsky and reduced Navier-Stokes. Ensemble SINDy bootstraps the trajectory and keeps only terms that survive 95% of resamples, which gives the error bars the basic method lacks. A Hamiltonian neural network can find the conserved structure first and then be distilled into a sparse polynomial for readability. PySR scales GP across GPUs and distributed populations, reaching Feynman equations up to nine variables in seconds. And Noether's theorem points at the most ambitious version: detect the symmetry of the vector field, then feed the matching conserved current back in as a hard constraint, so the model satisfies the physics exactly instead of approximately.
