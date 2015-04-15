Imagine you're a budding chef. A data-curious one, of course, so you start by taking a set of foods (pizza, salad, spaghetti, etc.) and ask 10 friends how much of each they ate in the past day.

Your goal: to find natural *groups* of foodies, so that you can better cater to each cluster's tastes. For example, your fratboy friends might love [wings and beer](https://twitter.com/#!/edchedch/status/166343879547822080), your anime friends might love soba and sushi, your hipster friends probably dig tofu, and so on.

So how can you use the data you've gathered to discover different kinds of groups?

[![Clustering Example](http://dl.dropbox.com/u/10506/blog/dirichlet-process/clustering-example.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/clustering-example.png)

One way is to use a standard clustering algorithm like **k-means** or **Gaussian mixture modeling** (see [this previous post](http://blog.echen.me/2011/03/19/counting-clusters/) for a brief introduction). The problem is that these both assume a *fixed* number of clusters, which they need to be told to find. There are a couple methods for selecting the number of clusters to learn (e.g., the [gap and prediction strength statistics](http://blog.echen.me/2011/03/19/counting-clusters/)), but the problem is a more fundamental one: most real-world data simply doesn't have a fixed number of clusters.

That is, suppose we've asked 10 of our friends what they ate in the past day, and we want to find groups of eating preferences. There's really an infinite number of foodie types (carnivore, vegan, snacker, Italian, healthy, fast food, heavy eaters, light eaters, and so on), but with only 10 friends, we simply don't have enough data to detect them all. (Indeed, we're limited to 10 clusters!) So whereas k-means starts with the incorrect assumption that there's a fixed, finite number of clusters that our points come from, *no matter if we feed it more data*, what we'd really like is a method positing an infinite number of hidden clusters that naturally arise as we ask more friends about their food habits. (For example, with only 2 data points, we might not be able to tell the difference between vegans and vegetarians, but with 200 data points, we probably could.)

Luckily for us, this is precisely the purview of **nonparametric Bayes**.*

*Nonparametric Bayes refers to a class of techniques that allow some parameters to change with the data. In our case, for example, instead of fixing the number of clusters to be discovered, we allow it to grow as more data comes in.

# A Generative Story

Let's describe a generative model for finding clusters in any set of data. We assume an infinite set of latent groups, where each group is described by some set of parameters. For example, each group could be a Gaussian with a specified mean `μ_i` and standard deviation `σ_i`, and these group parameters themselves are assumed to come from some base distribution `G_0`. Data is then generated in the following manner:

* Select a cluster.
* Sample from that cluster to generate a new point.

(Note the resemblance to a [finite mixture model](http://en.wikipedia.org/wiki/Mixture_model).)

For example, suppose we ask 10 friends how many calories of pizza, salad, and rice they ate yesterday. Our groups could be:

* A Gaussian centered at (pizza = 5000, salad = 100, rice = 500) (i.e., a pizza lovers group).
* A Gaussian centered at (pizza = 100, salad = 3000, rice = 1000) (maybe a vegan group).
* A Gaussian centered at (pizza = 100, salad = 100, rice = 10000) (definitely Asian).
* ...

When deciding what to eat when she woke up yesterday, Alice could have thought *girl, I'm in the mood for pizza* and her food consumption yesterday would have been a sample from the pizza Gaussian. Similarly, Bob could have spent the day in Chinatown, thereby sampling from the Asian Gaussian for his day's meals. And so on.

The big question, then, is: how do we assign each friend to a group?

# Assigning Groups

## Chinese Restaurant Process

One way to assign friends to groups is to use a **Chinese Restaurant Process**. This works as follows: Imagine a restaurant where all your friends went to eat yesterday...

* Initially the restaurant is empty.
* The first person to enter (Alice) sits down at a table (selects a group). She then orders food for the table (i.e., she selects parameters for the group); everyone else who joins the table will then be limited to eating from the food she ordered.
* The second person to enter (Bob) sits down at a table. Which table does he sit at? With probability `α / (1 + α)` he sits down at a new table (i.e., selects a new group) and orders food for the table; with probability `1 / (1 + α)` he sits with Alice and eats from the food she's already ordered (i.e., he's in the same group as Alice).
* ...
* The (n+1)-st person sits down at a new table with probability `α / (n + α)`, and at table k with probability `n_k / (n + α)`, where `n_k` is the number of people currently sitting at table k.

Note a couple things:

* The more people (data points) there are at a table (cluster), the more likely it is that people (new data points) will join it. In other words, our groups satisfy a **rich get richer** property.
* There's always a small probability that someone joins an entirely new table (i.e., a new group is formed).
* The probability of a new group depends on `α`. So we can think of `α` as a **dispersion parameter** that affects the dispersion of our datapoints. The lower alpha is, the more tightly clustered our data points; the higher it is, the more clusters we have in any finite set of points.

(Also notice the resemblance between table selection probabilities and a Dirichlet distribution...)

Just to summarize, given n data points, the Chinese Restaurant Process specifies a distribution over partitions (table assignments) of these points. We can also generate parameters for each partition/table from a base distribution `G_0` (for example, each table could represent a Gaussian whose mean and standard deviation are sampled from `G_0`), though to be clear, this is not part of the CRP itself.

### Code

Since code makes everything better, here's some Ruby to simulate a CRP:

``` ruby
# Generate table assignments for `num_customers` customers, according to
# a Chinese Restaurant Process with dispersion parameter `α`.
#
# returns an array of integer table assignments
def chinese_restaurant_process(num_customers, alpha)
 return [] if num_customers <= 0

 table_assignments = [1] # first customer sits at table 1
 next_open_table = 2 # index of the next empty table

 # Now generate table assignments for the rest of the customers.
 1.upto(num_customers - 1) do |i|
   if rand < alpha.to_f / (alpha + i)
     # Customer sits at new table.
     table_assignments << next_open_table
     next_open_table += 1
   else
     # Customer sits at an existing table.
     # He chooses which table to sit at by giving equal weight to each
     # customer already sitting at a table. 
     which_table = table_assignments[rand(table_assignments.size)]
     table_assignments << which_table
   end
 end

 table_assignments
end
```

And here's some sample output:

```
> chinese_restaurant_process(num_customers = 10, alpha = 1)
1, 2, 3, 4, 3, 3, 2, 1, 4, 3 # table assignments from run 1
1, 1, 1, 1, 1, 1, 2, 2, 1, 3 # table assignments from run 2
1, 2, 2, 1, 3, 3, 2, 1, 3, 4 # table assignments from run 3

> chinese_restaurant_process(num_customers = 10, alpha = 3)
1, 2, 1, 1, 3, 1, 2, 3, 4, 5
1, 2, 3, 3, 4, 3, 4, 4, 5, 5
1, 1, 2, 3, 1, 4, 4, 3, 1, 1

> chinese_restaurant_process(num_customers = 10, alpha = 5)
1, 2, 1, 3, 4, 5, 6, 7, 1, 8
1, 2, 3, 3, 4, 5, 6, 5, 6, 7
1, 2, 3, 4, 5, 6, 2, 7, 2, 1
```

Notice that as we increase `α`, so too does the number of distinct tables increase.

## Polya Urn Model

Another method for assigning friends to groups is to follow the **Polya Urn Model**. This is basically the same model as the Chinese Restaurant Process, just with a different metaphor.

* We start with an urn containing `α G_0(x)` balls of "color" `x`, for each possible value of `x`. (`G_0` is our base distribution, and `G_0(x)` is the probability of sampling `x` from `G_0`). Note that these are possibly fractional balls.
* At each time step, draw a ball from the urn, note its color, and then drop both the original ball plus a new ball of the same color back into the urn.

Note the connection between this process and the CRP: balls correspond to people (i.e., data points), colors correspond to table assignments (i.e., clusters), alpha is again a dispersion parameter (put differently, a prior), colors satisfy a rich-get-richer property (since colors with many balls are more likely to get drawn), and so on. (Again, there's also a connection between this urn model and [the urn model for the (finite) Dirichlet distribution](http://en.wikipedia.org/wiki/Dirichlet_distribution#P.C3.B3lya.27s_urn)...)

To be precise, the difference between the CRP and the Polya Urn Model is that the CRP specifies only a distribution over *partitions* (i.e., table assignments), but doesn't assign parameters to each group, whereas the Polya Urn Model does both.

### Code

Again, here's some code for simulating a Polya Urn Model:

``` ruby
# Draw `num_balls` colored balls according to a Polya Urn Model
# with a specified base color distribution and dispersion parameter
# `α`.
#
# returns an array of ball colors
def polya_urn_model(base_color_distribution, num_balls, alpha)
  return [] if num_balls <= 0

  balls_in_urn = []
  0.upto(num_balls - 1) do |i|
    if rand < alpha.to_f / (alpha + balls_in_urn.size)
      # Draw a new color, put a ball of this color in the urn.
      new_color = base_color_distribution.call      
      balls_in_urn << new_color
    else
      # Draw a ball from the urn, add another ball of the same color.
      ball = balls_in_urn[rand(balls_in_urn.size)]
      balls_in_urn << ball
    end
  end

  balls_in_urn
end
```

And here's some sample output, using a uniform distribution over the unit interval as the color distribution to sample from:

```
> unit_uniform = lambda { (rand * 100).to_i / 100.0 }

> polya_urn_model(unit_uniform, num_balls = 10, alpha = 1)
0.27, 0.89, 0.89, 0.89, 0.73, 0.98, 0.43, 0.98, 0.89, 0.53 # colors in the urn from run 1
0.26, 0.26, 0.46, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.85 # colors in the urn from run 2
0.96, 0.87, 0.96, 0.87, 0.96, 0.96, 0.87, 0.96, 0.96, 0.96 # colors in the urn from run 3
```

### Code, Take 2

Here's the same code for a Polya Urn Model, but in R:

``` r
# Return a vector of `num_balls` ball colors according to a Polya Urn Model
# with dispersion `α`, sampling from a specified base color distribution.
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
```

Here are some sample density plots of the colors in the urn, when using a unit normal as the base color distribution:

[![Polya Urn Model, Alpha = 1](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_1.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_1.png)

[![Polya Urn Model, Alpha = 5](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_5.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_5.png)

[![Polya Urn Model, Alpha = 25](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_25.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_25.png)

[![Polya Urn Model, Alpha = 50](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_50.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_50.png)

Notice that as alpha increases (i.e., we sample more new ball colors from our base; i.e., as we place more weight on our prior), the colors in the urn tend to a unit normal (our base color distribution).

And here are some sample plots of points generated by the urn, for varying values of alpha:

* Each color in the urn is sampled from a uniform distribution over \[0,10\] x \[0,10\] (i.e., a [0, 10] square).
* Each group is a Gaussian with standard deviation 0.1 and mean equal to its associated color, and these Gaussian groups generate points.

[![Alpha 0.1](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.1.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.1.png)

[![Alpha 0.2](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.2.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.2.png)

[![Alpha 0.3](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.3.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.3.png)

[![Alpha 0.5](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.5.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-0.5.png)

[![Alpha 1.0](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-1.0.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/alpha-1.0.png)

Notice that the points clump together in fewer clusters for low values of alpha, but become more dispersed as alpha increases. 

## Stick-Breaking Process

Imagine running either the Chinese Restaurant Process or the Polya Urn Model without stop. For each group `i`, this gives a proportion `w_i` of points that fall into group `i`.

So instead of running the CRP or Polya Urn model to figure out these proportions, can we simply generate them directly?

This is exactly what the Stick-Breaking Process does:

* Start with a stick of length one.
* Generate a random variable `β_1 ~ Beta(1, α)`. By the definition of the [Beta distribution](http://en.wikipedia.org/wiki/Beta_distribution), this will be a real number between 0 and 1, with expected value `1 / (1 + α)`. Break off the stick at `β_1`; `w_1` is then the length of the stick on the left.
* Now take the stick to the right, and generate `β_2 ~ Beta(1, α)`. Break off the stick `β_2` into the stick. Again, `w_2` is the length of the stick to the left, i.e., `w_2 = (1 - \beta_1) \beta_2`.
* And so on.

Thus, the Stick-Breaking process is simply the CRP or Polya Urn Model from a different point of view. For example, assigning customers to table 1 according to the Chinese Restaurant Process is equivalent to assigning customers to table 1 with probability `w_1`.

### Code

Here's some R code for simulating a Stick-Breaking process:

``` r
# Return a vector of weights drawn from a stick-breaking process
# with dispersion `α`.
#
# Recall that the kth weight is
#   \beta_k = (1 - \beta_1) * (1 - \beta_2) * ... * (1 - \beta_{k-1}) * beta_k
# where each `beta_i` is drawn from a Beta distribution
#   \beta_i ~ Beta(1, α)
stick_breaking_process = function(num_weights, alpha) {
  betas = rbeta(num_weights, 1, alpha)
  remaining_stick_lengths = c(1, cumprod(1 - betas))[1:num_weights]
  weights = remaining_stick_lengths * betas
  weights
}
```

And here's some sample output:

[![Stick-Breaking Process, alpha = 1](http://dl.dropbox.com/u/10506/blog/dirichlet-process/sbp_alpha_1.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/sbp_alpha_1.png)

[![Stick-Breaking Process, alpha = 3](http://dl.dropbox.com/u/10506/blog/dirichlet-process/sbp_alpha_3.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/sbp_alpha_3.png)

[![Stick-Breaking Process, alpha = 5](http://dl.dropbox.com/u/10506/blog/dirichlet-process/sbp_alpha_5.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/sbp_alpha_5.png)

Notice that for low values of alpha, the stick weights are concentrated on the first few weights (meaning our data points are concentrated on a few clusters), while the weights become more evenly dispersed as we increase alpha (meaning we posit more clusters in our data points).

## Dirichlet Process

Suppose we run a Polya Urn Model several times, where we sample colors from a base distribution `G_0`. Each run produces a distribution of colors in the urn (say, 5% blue balls, 3% red balls, 2% pink balls, etc.), and the distribution will be different each time (for example, 5% blue balls in run 1, but 1% blue balls in run 2).

For example, let's look again at the plots from above, where I generated samples from a Polya Urn Model with the standard unit normal as the base distribution:

[![Polya Urn Model, Alpha = 1](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_1.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_1.png)

[![Polya Urn Model, Alpha = 5](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_5.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_5.png)

[![Polya Urn Model, Alpha = 25](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_25.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_25.png)

[![Polya Urn Model, Alpha = 50](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_50.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/polya_alpha_50.png)

Each run of the Polya Urn Model produces a slighly different distribution, though each is "centered" in some fashion around the standard Gaussian I used as base. In other words, the Polya Urn Model gives us a **distribution over distributions** (we get a distribution of ball colors, and this distribution of colors changes each time) -- and so we finally get to the Dirichlet Process.

Formally, given a base distribution `G_0` and a dispersion parameter `α`, a sample from the Dirichlet Process `DP(G_0, α)` is a distribution `G ~ DP(G_0, α)`. This sample `G` can be thought of as a distribution of colors in a single simulation of the Polya Urn Model; sampling from `G` gives us the balls in the urn.

So here's the connection between the Chinese Restaurant Process, the Polya Urn Model, the Stick-Breaking Process, and the Dirichlet Process:

* **Dirichlet Process**: Suppose we want samples `x_i ~ G`, where `G` is a distribution sampled from the Dirichlet Process `G ~ DP(G_0, α)`.
* **Polya Urn Model**: One way to generate these values `x_i` would be to take a Polya Urn Model with color distribution `G_0` and dispersion `α`. (`x_i` would be the color of the ith ball in the urn.)
* **Chinese Restaurant Process**: Another way to generate `x_i` would be to first assign tables to customers according to a Chinese Restaurant Process with dispersion `α`. Every customer at the nth table would then be given the same value (color) sampled from `G_0`. (`x_i` would be the value given to the ith customer; `x_i` can also be thought of as the food at table `i`, or as the parameters of table `i`.)
* **Stick-Breaking Process**: Finally, we could generate weights `w_k` according to a Stick-Breaking Process with dispersion `α`. Next, we would give each weight `w_k` a value (or color) `v_k` sampled from `G_0`. Finally, we would assign `x_i` to value (color) `v_k` with probability `w_k`.

# Recap

Let's summarize what we've discussed so far.

We have a bunch of data points `p_i` that we want to cluster, and we've described four essentially equivalent generative models that allow us to describe how each cluster and point could have arisen.

In the **Chinese Restaurant Process**:

* We generate table assignments `g_1, ..., g_n ~ CRP(α)` according to a Chinese Restaurant Process. (`g_i` is the table assigned to datapoint `i`.)
* We generate table parameters `φ_1, ..., φ_m ~ G_0` according to the base distribution `G_0`, where `φ_k` is the parameter for the kth distinct group.
* Given table assignments and table parameters, we generate each datapoint `p_i ~ F(φ_{g_i})` from a distribution `F` with the specified table parameters. (For example, `F` could be a Gaussian, and `φ_i` could be a parameter vector specifying the mean and standard deviation).

In the **Polya Urn Model**:

* We generate colors `φ_1, ..., φ_n ~ Polya(G_0, α)` according to a Polya Urn Model. (`φ_i` is the color of the ith ball.)
* Given ball colors, we generate each datapoint `p_i ~ F(φ_i)`.

In the **Stick-Breaking Process**:

* We generate group probabilities (stick lengths) `w_1, ..., w_{∞} ~ Stick(α)` according to a Stick-Breaking process.
* We generate group parameters `φ_1, ..., φ_{∞} ~ G_0` from `G_0`, where `φ_k` is the parameter for the kth distinct group.
* We generate group assignments `g_1, ..., g_n ~ Multinomial(w_1, ..., w_{∞})` for each datapoint.
* Given group assignments and group parameters, we generate each datapoint `p_i ~ F(φ_{g_i})`.

In the **Dirichlet Process**:

* We generate a distribution `G ~ DP(G_0, α)` from a Dirichlet Process with base distribution `G_0` and dispersion parameter `α`.
* We generate group-level parameters `x_i ~ G` from `G`, where `x_i` is the group parameter for the ith datapoint. (Note: this is not the same as `φ_i`. `x_i` is the parameter associated to the group that the ith datapoint belongs to, whereas `φ_k` is the parameter of the kth distinct group.)
* Given group-level parameters `x_i`, we generate each datapoint `p_i ~ F(x_i)`.

Also, remember that each model naturally allows the number of clusters to grow as more points come in.

# Inference in the Dirichlet Process Mixture

So we've described a generative model that allows us to calculate the probability of any particular set of group assignments to data points, but we haven't described how to actually learn a good set of group assignments.

Let's briefly do this now. Very roughly, the **Gibbs sampling** approach works as follows:

* Take the set of data points, and randomly initialize group assignments.
* Pick a point. Fix the group assignments of all the other points, and assign the chosen point a new group (which can be either an existing cluster or a new cluster) with a CRP-ish probability (as described in the models above) that depends on the group assignments and values of all the other points.
* We will eventually converge on a good set of group assignments, so repeat the previous step until happy.

For more details, [this paper](http://www.cs.toronto.edu/~radford/ftp/mixmc.pdf) provides a good description. Philip Resnick and Eric Hardisty also have a friendlier, more general description of Gibbs sampling (plus an application to naive Bayes) [here](http://www.cs.umd.edu/~hardisty/papers/gsfu.pdf).

# Fast Food Application: Clustering the McDonald's Menu

Finally, let's show an application of the Dirichlet Process Mixture. Unfortunately, I didn't have a data set of people's food habits offhand, so instead I took [this list](http://nutrition.mcdonalds.com/nutritionexchange/nutritionfacts.pdf) of McDonald's foods and nutrition facts.

After normalizing each item to have an equal number of calories, and representing each item as a vector of **(total fat, cholesterol, sodium, dietary fiber, sugars, protein, vitamin A, vitamin C, calcium, iron, calories from fat, satured fat, trans fat, carbohydrates)**, I ran [scikit-learn](http://scikit-learn.sourceforge.net/dev/index.html)'s [Dirichlet Process Gaussian Mixture Model](http://scikit-learn.sourceforge.net/dev/modules/mixture.html) to cluster McDonald's menu based on nutritional value.

First, how does the number of clusters inferred by the Dirichlet Process mixture vary as we feed in more (randomly ordered) points?

[![Growth of Number of Clusters](http://dl.dropbox.com/u/10506/blog/dirichlet-process/num-clusters-vary.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/num-clusters-vary.png)

As expected, the Dirichlet Process model discovers more and more clusters as more and more food items arrive. (And indeed, the number of clusters appears to grow logarithmically, which can in fact be proved.)

How many clusters does the mixture model infer from the entire dataset? Running the Gibbs sampler several times, we find that the number of clusters tends around 11:

[![Number of clusters](http://dl.dropbox.com/u/10506/blog/dirichlet-process/num_mcdonalds_clusters_small.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/num_mcdonalds_clusters.png)

Let's dive into one of these clusterings.

**Cluster 1 (Desserts)**

Looking at a sample of foods from the first cluster, we find a lot of desserts and dessert-y drinks:

* Caramel Mocha
* Frappe Caramel
* Iced Hazelnut Latte
* Iced Coffee
* Strawberry Triple Thick Shake
* Snack Size McFlurry
* Hot Caramel Sundae
* Baked Hot Apple Pie
* Cinnamon Melts
* Kiddie Cone
* Strawberry Sundae

We can also look at the nutritional profile of some foods from this cluster (after [z-scaling](http://en.wikipedia.org/wiki/Standard_score) each nutrition dimension to have mean 0 and standard deviation 1):

[![Cluster 1](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster1.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster1.png)

We see that foods in this cluster tend to be high in trans fat and low in vitamins, protein, fiber, and sodium.

**Cluster 2 (Sauces)**

Here's a sample from the second cluster, which contains a lot of sauces:

* Hot Mustard Sauce
* Spicy Buffalo Sauce
* Newman's Own Low Fat Balsamic Vinaigrette

And looking at the nutritional profile of points in this cluster, we see that it's heavy in sodium and fat:

[![Cluster 2](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster2.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster2.png)

**Cluster 3 (Burgers, Crispy Foods, High-Cholesterol)**

The third cluster is very burgery:

* Hamburger
* Cheeseburger
* Filet-O-Fish
* Quarter Pounder with Cheese
* Premium Grilled Chicken Club Sandwich
* Ranch Snack Wrap
* Premium Asian Salad with Crispy Chicken
* Butter Garlic Croutons
* Sausage McMuffin
* Sausage McGriddles

It's also high in fat and sodium, and low in carbs and sugar

[![Cluster 3](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster3.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster3.png)

**Cluster 4 (Creamy Sauces)**

Interestingly, even though we already found a cluster of sauces above, we discover another one as well. These sauces appear to be much more cream-based:

* Creamy Ranch Sauce
* Newman's Own Creamy Caesar Dressing
* Coffee Cream
* Iced Coffee with Sugar Free Vanilla Syrup

Nutritionally, these sauces are higher in calories from fat, and much lower in sodium:

[![Cluster 4](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster4.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster4.png)

**Cluster 5 (Salads)**

Here's a salad cluster. A lot of salads also appeared in the third cluster (along with hamburgers and McMuffins), but that's because those salads also all contained crispy chicken. The salads in this cluster are either crisp-free or have their chicken grilled instead:

* Premium Southwest Salad with Grilled Chicken
* Premium Caesar Salad with Grilled Chicken
* Side Salad
* Premium Asian Salad without Chicken
* Premium Bacon Ranch Salad without Chicken

This is reflected in the higher content of iron, vitamin A, and fiber:

[![Cluster 5](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster5.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster5.png)

**Cluster 6 (More Sauces)**

Again, we find another cluster of sauces:

* Ketchup Packet
* Barbeque Sauce
* Chipotle Barbeque Sauce

These are still high in sodium, but much lower in fat compared to the other sauce clusters:

[![Cluster 6](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster6.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster6.png)

**Cluster 7 (Fruit and Maple Oatmeal)**

Amusingly, fruit and maple oatmeal is in a cluster by itself:

* Fruit & Maple Oatmeal

[![Cluster 7](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster7.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster7.png)

**Cluster 8 (Sugary Drinks)**

We also get a cluster of sugary drinks:

* Strawberry Banana Smoothie
* Wild Berry Smoothie
* Iced Nonfat Vanilla Latte
* Nonfat Hazelnut
* Nonfat Vanilla Cappuccino
* Nonfat Caramel Cappuccino
* Sweet Tea
* Frozen Strawberry Lemonade
* Coca-Cola
* Minute Maid Orange Juice

In addition to high sugar content, this cluster is also high in carbohydrates and calcium, and low in fat.

[![Cluster 8](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster8.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster8.png)

**Cluster 9 (Breakfast Foods)**

Here's a cluster of high-cholesterol breakfast foods:

* Sausage McMuffin with Egg
* Sausage Burrito
* Egg McMuffin
* Bacon, Egg & Chees Biscuit
* McSkillet Burrito with Sausage
* Big Breakfast with Hotcakes

[![Cluster 9](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster9.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster9.png)

**Cluster 10 (Coffee Drinks)**

We find a group of coffee drinks next:

* Nonfat Cappuccino
* Nonfat Latte
* Nonfat Latte with Sugar Free Vanilla Syrup
* Iced Nonfat Latte

These are much higher in calcium and protein, and lower in sugar, than the other drink cluster above:

[![Cluster 11](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster11.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster11.png)

**Cluster 11 (Apples)**

Here's a cluster of apples:

* Apple Dippers with Low Fat Caramel Dip
* Apple Slices

Vitamin C, check.

[![Cluster 10](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster10.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/cluster10.png)

And finally, here's an overview of all the clusters at once (using a different clustering run):

[![All Clusters](http://dl.dropbox.com/u/10506/blog/dirichlet-process/all-clusters-small.png)](http://dl.dropbox.com/u/10506/blog/dirichlet-process/all-clusters.png)

# No More!

I'll end with a couple notes:

* Kevin Knight has a [hilarious introduction](http://www.isi.edu/natural-language/people/bayes-with-tears.pdf) to Bayesian inference that describes some applications of nonparametric Bayesian techniques to computational linguistics (though I don't think he ever quite says "nonparametric Bayes" directly).
* In the Chinese Restaurant Process, each customer sits at a single table. The [Indian Buffet Process](http://en.wikipedia.org/wiki/Chinese_restaurant_process#The_Indian_buffet_process) is an extension that allows customers to sample food from multiple tables (i.e., belong to multiple clusters).
* The Chinese Restaurant Process, the Polya Urn Model, and the Stick-Breaking Process are all *sequential* models for generating groups: to figure out table parameters in the CRP, for example, you wait for customer 1 to come in, then customer 2, then customer 3, and so on. The equivalent Dirichlet Process, on the other hand, is a *parallelizable* model for generating groups: just sample `G ~ DP(G_0, α)`, and then all your group parameters can be independently generated by sampling from `G` at once. This duality is an instance of a more general phenomenon known as [de Finetti's theorem](http://en.wikipedia.org/wiki/De_Finetti's_theorem).

And that's it.
