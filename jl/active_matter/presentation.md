# Active Matter: Collective Motion and Single-Cell Swimming

Parth Bhargava · computational physics

Two models, both written from scratch in Julia. One is about a crowd, the other about a single cell. Both are "active" in the technical sense, and that one word ends up carrying most of the physics. Each numbered section below is a slide; the indented *Animation* line is the clip that plays on it (all rendered in `animations/output/`).

---

## Outline

1. The hook: order with nobody in charge
2. What "active" actually buys you
3. The Vicsek model
4. Measuring alignment with one number
5. The flocking transition
6. Run-and-tumble: a single bacterium
7. Why the motion goes ballistic, then diffusive
8. Pinning down the diffusion constant
9. The interactive notebook
10. Where this goes next

---

## 1. The hook: order with nobody in charge

Starlings turn together. A flock of a few thousand banks and folds as if it were a single object, and the turn sweeps across the whole flock in well under a second. No lead bird is calling it. Each one is just watching the handful of neighbours nearest to it and matching their direction.

The opening slide is the barest version of that idea. Same particles, same rule (match your neighbours' heading, then add a random kick), run once with the kick small and once with it large. Small kick: they collapse into one streaming flock. Large kick: they drift around with no shared direction at all. Nobody programmed the flock in. It appears on its own once the noise drops below some threshold, and finding that threshold is what the rest of this is about.

> **Animation:** `s02_vicsek_phases.mp4`. The disordered gas (η=3) shows up first, milling around with φ≈0. Then the same model at η=0.5 fades in next to it and snaps into a flock, φ climbing to about 0.87. The caption makes the point: identical rule, the noise is the only thing that changed.

---

## 2. What "active" actually buys you

Drop a pollen grain in water and the molecules hammering it keep it jittering, but it never picks a direction. Over time it just diffuses. The kicks pushing it and the drag slowing it are two faces of the same thermal coin, which is the fluctuation–dissipation theorem and, really, the definition of equilibrium.

An active particle carries its own motor. A bacterium burns ATP; a Janus colloid runs a chemical reaction on one face. Either way it propels itself along a direction it sets, and it keeps doing so. The instant self-propulsion enters at the single-particle level, the thermal bookkeeping stops balancing. Detailed balance is gone, and the equilibrium toolkit goes with it.

That sounds abstract until you see what it lets the system do that a passive one simply can't:

| | Passive (thermal) | Active |
|---|---|---|
| Net current at steady state | none | allowed |
| Long-range order in 2D | forbidden (Mermin–Wagner) | allowed (Toner–Tu) |
| What tunes a transition | temperature | density and noise |
| Clustering | needs an attractive force | happens from motility alone |

Every result later in the talk traces back to that one move: energy going in one particle at a time.

> **Animation:** `s01_motivation.mp4`. Passive particles diffusing on the spot with ⟨v⟩=0 on the left, active particles streaming and clumping on the right. Same count of particles. Only the active ones have a motor.

---

## 3. The Vicsek model

Vicsek's 1995 model is the stripped-down recipe for a flock. Keep cutting ingredients until two are left: self-propulsion at a fixed speed, and a tendency to align with whoever is nearby.

At each step a particle looks at every neighbour inside a radius $r$ (counting itself), takes their average heading, and adds a random kick:

$$\theta_i(t+1) = \Big\langle \theta_j(t) \Big\rangle_{|\mathbf r_j-\mathbf r_i|<r} + \xi_i,\qquad \xi_i\sim\text{Uniform}\!\left(-\tfrac{\eta}{2},\tfrac{\eta}{2}\right)$$

$$\mathbf r_i(t+1) = \mathbf r_i(t) + v_0\,(\cos\theta_i,\ \sin\theta_i)$$

There's one trap in the implementation. You can't average angles as plain numbers, because the mean of 1° and 359° comes out as 180° when the honest answer is 0°. So the code averages the unit vectors instead and reads off the angle, $\arg\sum_j e^{i\theta_j}$. With that fixed, the model is complete. Alignment pushes the system toward order, the noise η drags it back toward disorder, and where they balance is the interesting question.

---

## 4. Measuring alignment with one number

To get a measurement out of a movie I need a single number that says how aligned the flock is. The obvious one is the length of the average heading vector:

$$\varphi = \frac{1}{N}\left|\sum_{i=1}^{N} e^{i\theta_i}\right| \in [0,1]$$

If every particle points the same way, φ sits at 1. If the headings are random, φ lands around $1/\sqrt N$, which is just what you get from adding N random unit vectors. It's the same quantity as the magnetisation of a ferromagnet, with headings standing in for spins.

The time trace of φ already tells you which phase you're in. At low noise it climbs toward 1 and stays there. Crank the noise to the critical value and it thrashes around. Push it higher and it sags back down to that random-phase floor.

> **Animation:** `s03_order_parameter.mp4`. The three φ(t) traces draw on one at a time, each captioned as it arrives: η=0.5 locks in near 1, η≈η_c rattles around mid-range, η=3.0 decays to the 1/√N line. The averaged curve ⟨φ⟩(η) then builds itself on the right panel.

---

## 5. The flocking transition

Average φ over time, sweep the noise, and plot ⟨φ⟩ against η. What comes out is a real phase transition: order holds below a critical noise $\eta_c$ and dies above it. The giveaway is the fluctuations. The susceptibility $\chi = N(\langle\varphi^2\rangle-\langle\varphi\rangle^2)$ spikes right at $\eta_c$, which is the finite-system version of a diverging response.

Density is what sets where $\eta_c$ falls. Pack particles closer and each one has more neighbours inside its radius, so alignment carries more weight and order survives to a higher noise. The scaling is $\eta_c \propto \sqrt{\rho}$ in two dimensions, and you can watch the whole curve slide right by roughly √2 when you double the density.

This is the part that makes active matter its own field rather than a footnote to thermodynamics. There is no temperature in the problem at all, and the ordered state itself is one that Mermin–Wagner forbids at equilibrium in 2D. Toner and Tu showed why activity gets around that.

> **Animation:** `s04_bifurcation.mp4`. The two bifurcation curves sweep in (ρ=1.5, then ρ=3.0), the predicted η_c markers drop in, and the susceptibility panel fills in its peak. The denser flock holds out to a higher noise, and you can see it.

---

## 6. Run-and-tumble: a single bacterium

The second model drops the crowd and follows one swimmer, and it's about the simplest honest description of how *E. coli* gets around. The cell does two things in alternation. It runs, swimming roughly straight, and now and then it tumbles, its flagella coming apart so it spins to a new random heading. Tumbles hit as a Poisson process at rate λ:

- **run:** keep moving at speed $v$ along the current heading
- **tumble:** with probability $\lambda\,\Delta t$ each step, pick a brand-new heading at random

So a run lasts about $\tau=1/\lambda$ on average and covers a length $\ell = v\tau$. That τ is the cell's memory: how long it holds a direction before forgetting it. Real bacteria do chemotaxis by tuning λ to the local chemical gradient, but the constant-λ version already has the transport physics in it, so that's what I simulate.

> **Animation:** `s05_runtumble_path.mp4`. One trajectory draws itself out. Every tumble flashes a marker and the heading θ(t) jumps in the side panel, so the "straight run, sudden kink, straight run" rhythm is right there. The run length ℓ = v/λ is marked on one segment.

---

## 7. Why the motion goes ballistic, then diffusive

Run-and-tumble is one of the few active models you can solve with pencil and paper, so it's worth doing. The velocity stays correlated only until the next tumble, which gives $\langle\mathbf v(t)\cdot\mathbf v(0)\rangle = v^2 e^{-\lambda t}$ (the odds of surviving to time $t$ without a tumble are $e^{-\lambda t}$). Integrate that correlation twice and the mean-squared displacement falls out exactly:

$$\langle r^2(t)\rangle = \frac{v^2}{\lambda^2}\Big[\,2\lambda t - 2\big(1-e^{-\lambda t}\big)\Big]$$

Look at the two ends. For times shorter than a run, $t\ll\tau$, this is $\langle r^2\rangle \approx v^2 t^2$: the cell is flying straight, so displacement grows linearly in time and $r^2$ grows like $t^2$. For times much longer than a run, $t\gg\tau$, it's $\langle r^2\rangle \approx 4 D_\text{eff}\,t$: enough tumbles have happened that the memory is wiped and the motion is a random walk. The handover sits at one timescale, $t\sim\tau$. That's the whole reason a bacterium reads as a tiny rocket up close and a diffusing speck from far away.

> **Animation:** `s06_msd.mp4`. The two scaling laws (∼t² and ∼t) get drawn first, like a pair of rulers. Then the measured MSD draws over them and you watch it ride the t² line, bend at the marked t=τ, and settle onto the t line.

---

## 8. Pinning down the diffusion constant

At long times everything collapses onto one number, the effective diffusion constant $D_\text{eff} = v^2\tau/2 = v^2/(2\lambda)$. Tumble less often, run longer, spread faster. The reason this matters is the size of it: $D_\text{eff}$ comes from activity, and it's far larger than the thermal diffusion the same particle would have if it were dead. That gap is the quantitative fingerprint of being out of equilibrium.

For the check, I fit $D_\text{eff}$ from the diffusive tail of the simulated MSD at a range of tumble rates and lay the results against $v^2/(2\lambda)$. The measured points sit on the predicted curve to within a few percent across the whole range, so the simulation and the exact theory agree.

> **Animation:** `s07_deff_comparison.mp4`. The law $v^2/(2\lambda)$ draws as a smooth curve, then each measured point drops onto it carrying its percent agreement. The side panel shows the run time τ=1/λ stretching out as the tumbling slows.

---

## 9. The interactive notebook

The Pluto notebook runs both models live with sliders, so the room can poke at the physics instead of watching a fixed clip.

On the Vicsek side: N particles coloured by heading, with φ(t) ticking along underneath. Sliders for N, η, density, and speed. The one experiment I always do live is to ease η down through η_c and watch a directionless gas pick a heading and start streaming. Symmetry breaking, happening while you talk.

On the run-and-tumble side: a cloud of particles trailing faint tracks, with the MSD redrawing every few steps on top of the exact curve, so the ballistic-to-diffusive bend appears as you watch it.

> **Visual:** screenshot of `app.pluto.jl`. The two-panel notebook, Vicsek flock on top, run-and-tumble MSD below.

---

## 10. Where this goes next

What's here: a Vicsek flocker and a run-and-tumble swimmer, both pure Julia, both fast enough to drive interactively, and both pinned to a textbook result with numbers rather than hand-waving.

| Model | Result | How it was checked |
|---|---|---|
| Vicsek | noise-driven flocking transition | $\eta_c\propto\sqrt\rho$; susceptibility peaks at $\eta_c$ |
| Vicsek | spontaneous symmetry breaking | heading set by a fluctuation, not the start state |
| Run-and-tumble | ballistic-to-diffusive crossover | $\langle r^2\rangle$ bends from $t^2$ to $t$ at $t=\tau$ |
| Run-and-tumble | activity-boosted diffusion | $D_\text{eff}=v^2/2\lambda$, matched to a few percent |

The obvious next steps build straight on these two engines. Toner–Tu hydrodynamics explains why polar order is even allowed in 2D once a system is active. Motility-induced phase separation gives you clustering with no attraction in the model at all. Chemotaxis is run-and-tumble with λ tied to a gradient, and in the continuum limit it becomes Keller–Segel. And at high density and low noise the Vicsek flock goes turbulent, with an energy spectrum that isn't the Kolmogorov one. Any of them is a reasonable place to take this.
