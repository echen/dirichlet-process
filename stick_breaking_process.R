# Return a vector of weights drawn from a stick-breaking process
# with dispersion `alpha`.
#
# Recall that the kth weight is
#   \beta_k = (1 - \beta_1) * (1 - \beta_2) * ... * (1 - \beta_{k-1}) * beta_k
# where each $\beta_i$ is drawn from a Beta distribution
#   \beta_i ~ Beta(1, \alpha)
#
# Examples
#
#   stick_breaking_process(num_weight = 5, alpha = 1)
#     => c(0.712148550, 0.169208000, 0.101483441, 0.014156001, 0.001498306)
#
stick_breaking_process = function(num_weights, alpha) {
  betas = rbeta(num_weights, 1, alpha)
  remaining_stick_lengths = c(1, cumprod(1 - betas))[1:num_weights]
  weights = remaining_stick_lengths * betas
  weights
}