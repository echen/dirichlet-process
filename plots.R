library(ggplot2)
library(reshape)

# Some of the plots used in the blog post.

##########
# POLYA URN MODEL
##########

polya_urn_model_plots = function(num_balls, alpha) {
  # Lazy man's repetition...
  x1 = polya_urn_model(function() rnorm(1), num_balls, alpha)
  x2 = polya_urn_model(function() rnorm(1), num_balls, alpha)
  x3 = polya_urn_model(function() rnorm(1), num_balls, alpha)
  x4 = polya_urn_model(function() rnorm(1), num_balls, alpha)
  x5 = polya_urn_model(function() rnorm(1), num_balls, alpha)
  
  d1 = data.frame(x = x1, type = "run #1")
  d2 = data.frame(x = x2, type = "run #2")
  d3 = data.frame(x = x3, type = "run #3")
  d4 = data.frame(x = x4, type = "run #4")
  d5 = data.frame(x = x5, type = "run #5")
  d = rbind(d1, d2, d3, d4, d5)
  
  qplot(x = x, data = d, geom = "density", fill = 1, alpha = I(0.85), xlab = "Color of ball in urn", ylab = "Density", main = paste("Polya Urn Model with Gaussian colors and alpha =", alpha)) + facet_grid( . ~ type )
}

polya_urn_model_plots(10, 1)

##########
# STICK-BREAKING PROCESS
##########

stick_breaking_process_plots = function(num_weights, alpha) {
  x1 = stick_breaking_process(num_weights, alpha)
  x2 = stick_breaking_process(num_weights, alpha)
  x3 = stick_breaking_process(num_weights, alpha)
  x4 = stick_breaking_process(num_weights, alpha)
  x5 = stick_breaking_process(num_weights, alpha)
  
  d1 = data.frame(x = 1:num_weights, weight = x1, type = "run #1")
  d2 = data.frame(x = 1:num_weights, weight = x2, type = "run #2")
  d3 = data.frame(x = 1:num_weights, weight = x3, type = "run #3")
  d4 = data.frame(x = 1:num_weights, weight = x4, type = "run #4")
  d5 = data.frame(x = 1:num_weights, weight = x5, type = "run #5")        
  d = rbind(d1, d2, d3, d4, d5)
  
  qplot(x = x, weight = weight ,data = d, geom = "bar", xlab = "Stick", ylab = "Weight", main = paste("Stick-Breaking Process with alpha =", alpha), ylim = c(0, 1)) + scale_x_continuous(breaks = 1:num_weights) + facet_grid( . ~ type )
}

stick_breaking_process_plots(10, 5)

##########
# ALL CLUSTERS
##########

x = read.table("mcdonalds-data-with-clusters.tsv", header = T, sep = " ", comment.char = "", quote = "")

# Ignore duplicate food items.
x = ddply(x, .(name), function(df) head(df, 1))

# For each cluster, take at most 5 items (to avoid the plot being dominated by large clusters).
x = ddply(x, .(cluster), function(df) head(df, 5))

# Reorder names by cluster (so we can get a plot where all points in a cluster are together).
x$name = factor(x$name, levels = x$name[order(x$cluster)], ordered = T)

# Turn this into a tall-thin matrix.
m = melt(x, id = c("name", "cluster"))

qplot(variable, weight = value, data = m, fill = cluster, geom = "bar", xlab = "Nutritional variable", ylab = "z-scaled value", main = "McDonald's Food Clusters") + facet_wrap(~ name, ncol = 5) + coord_flip() + opts(axis.text.y = theme_text(size = 5), axis.text.x = theme_text(size = 5))