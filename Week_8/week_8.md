# Week 8 Assignment

## 1. Rewriting the sliced score matching loss

Let \(x \sim p(x)\) denote data samples and let \(v \sim p(v)\) be an isotropic direction (e.g., a standard normal that is later normalised). The sliced score matching (SSM) objective is defined as
\[
L_{SSM}
= \frac{1}{2}\,\mathbb{E}_{x,v}\!\left[\big(v^\top S(x;\theta) - v^\top \nabla_x \log p(x)\big)^2\right],
\]
where \(S(x;\theta)\) is the modelled score. Expanding the square gives
\[
L_{SSM}
= \frac{1}{2}\,\mathbb{E}_{x,v}\!\left[\|v^\top S(x;\theta)\|^2
- 2\,v^\top S(x;\theta)\, v^\top \nabla_x \log p(x)
+ \|v^\top \nabla_x \log p(x)\|^2\right].
\]
The last term is independent of \(\theta\) and can be treated as a constant. For the middle term we integrate by parts:
\[
\begin{aligned}
\mathbb{E}_{x,v}\!\left[v^\top S(x;\theta)\, v^\top \nabla_x \log p(x)\right]
&= \int p(x)\,\mathbb{E}_{v}\!\left[v^\top S(x;\theta)\, \frac{v^\top \nabla_x p(x)}{p(x)}\right]\mathrm{d}x \\
&= \int \mathbb{E}_{v}\!\left[v^\top S(x;\theta)\, v^\top \nabla_x p(x)\right]\mathrm{d}x \\
&= - \int p(x)\,\mathbb{E}_{v}\!\left[v^\top \nabla_x (v^\top S(x;\theta))\right]\mathrm{d}x,
\end{aligned}
\]
where we assume the boundary term vanishes because \(p(x)\) decays sufficiently fast. Substituting this back gives
\[
L_{SSM}= \frac{1}{2}\,\mathbb{E}_{x,v}\!\left[\|v^\top S(x;\theta)\|^2 + 2\, v^\top \nabla_x \big(v^\top S(x;\theta)\big)\right] + \text{const}.
\]
Since the additive constant and the positive scaling factor \(1/2\) do not affect the minimiser, we can equivalently write the loss as
\[
L_{SSM}= \mathbb{E}_{x\sim p(x)} \mathbb{E}_{v\sim p(v)} \left[\|v^\top S(x;\theta)\|^2+2\,v^\top\nabla_x \big(v^\top S(x;\theta)\big)\right].
\]

## 2. Brief explanation of SDEs

A stochastic differential equation (SDE) describes the evolution of a random process whose dynamics combine deterministic drift and stochastic noise:
\[
\mathrm{d}x_t = f(x_t, t)\,\mathrm{d}t + g(x_t, t)\,\mathrm{d}W_t,
\]
where \(f\) is the drift, \(g\) scales the noise, and \(W_t\) is a Wiener process. In diffusion models we pick \(f\) and \(g\) so that data points gradually diffuse toward a simple prior (the forward SDE), while the learned score allows us to integrate the corresponding reverse-time SDE to map noise back to data, enabling generative sampling with controllable randomness.

## 3. Unanswered questions from the lecture

- How sensitive is the reverse-time SDE to errors in the learned score, and are there diagnostics to detect instability early during training?
- What practical criteria guide the choice between Itô and Stratonovich interpretations when implementing diffusion SDEs in code?
- Are there principled ways to couple the noise schedule of the forward SDE with architectural choices (e.g., U-Net depth) to balance sample quality and training stability?
