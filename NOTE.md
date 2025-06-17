# Some emperical results

- From the DDPM paper, the authors found that: "Experimentally, both $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ had similar results."; however, from my experiments, I found that $\sigma_t^2 = \beta_t$ is better than $\sigma_t^2 = \tilde{\beta}_t$. (see he results in assets/assignment_1_task_1/)

