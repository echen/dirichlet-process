# Return a vector of weights drawn from a stick-breaking process
# with dispersion `alpha`.
#
# Recall that the kth weight is
#   \beta_k = (1 - \beta_1) * (1 - \beta_2) * ... * (1 - \beta_{k-1}) * beta_k
# where each $\beta_i$ is drawn from a Beta distribution
#   \beta_i ~ Beta(1, \alpha)
stick_breaking_process = function(num_weights, alpha) {
  betas = rbeta(num_weights, 1, alpha)
  remaining_stick_lengths = c(1, cumprod(1 - betas))[1:num_weights]
  weights = remaining_stick_lengths * betas
  weights
}