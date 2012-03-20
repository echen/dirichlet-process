# Generate table assignments for `num_customers` customers, according to
# a Chinese Restaurant Process with dispersion parameter `alpha`.
#
# returns an array of integer table assignments
def chinese_restaurant_process(num_customers, alpha)
  return [] if num_customers <= 0

  table_assignments = [1] # first customer sits at table 1
  next_open_table = 2 # index of the next empty table

  # Now generate table assignments for the rest of the customers.
  1.upto(num_customers - 1) do |i|
    if rand < alpha.to_f / (alpha + i)
      # Customer sits at a new table.
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

puts chinese_restaurant_process(num_customers = 10, alpha = 3).join(", ")