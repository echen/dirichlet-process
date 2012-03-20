# Return a vector of `num_balls` ball colors according to a Polya Urn Model
# with dispersion `alpha`, sampling from a specified base color distribution.
polya_urn_model = function(base_color_distribution, num_balls, alpha) {
  balls = c()
  
  for (i in 1:num_balls) {
    if (runif(1) < alpha / (alpha + length(balls))) {
      # Add a new ball color.
      new_color = base_color_distribution()
      balls = c(balls, new_color)
    } else {
      # Pick out a ball from the urn, and add back a
      # ball of the same color.
      ball = balls[sample(1:length(balls), 1)]
      balls = c(balls, ball)
    }
  }
  
  balls
}

# Sample run, using the unit Gaussian as the base color distribution.
polya_urn_model(function() rnorm(1), 100, 1)