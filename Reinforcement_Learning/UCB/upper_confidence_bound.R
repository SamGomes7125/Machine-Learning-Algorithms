# Upper Confidence Bound (UCB) Algorithm in R

# Importing the dataset
dataset <- read.csv('Ads_CTR_Optimisation.csv')

# Parameters
N <- 10000  # Number of rounds (users)
d <- 10     # Number of ads
ads_selected <- integer(0)  # Vector to store selected ads
numbers_of_selections <- numeric(d)  # Count of selections for each ad
sums_of_rewards <- numeric(d)  # Sum of rewards for each ad
total_reward <- 0

# UCB Algorithm
for (n in 1:N) {
  ad <- 1  # Start indexing at 1
  max_upper_bound <- 0
  
  for (i in 1:d) {
    if (numbers_of_selections[i] > 0) {
      average_reward <- sums_of_rewards[i] / numbers_of_selections[i]
      delta_i <- sqrt((3/2) * log(n) / numbers_of_selections[i])
      upper_bound <- average_reward + delta_i
    } else {
      upper_bound <- Inf  # Assign a very large value for first-time selections
    }
    
    if (upper_bound > max_upper_bound) {
      max_upper_bound <- upper_bound
      ad <- i
    }
  }
  
  # Select ad and update variables
  ads_selected <- c(ads_selected, ad)
  numbers_of_selections[ad] <- numbers_of_selections[ad] + 1
  reward <- dataset[n, ad]
  sums_of_rewards[ad] <- sums_of_rewards[ad] + reward
  total_reward <- total_reward + reward
}

# Visualizing the Results
hist(ads_selected, col = 'blue', main = "Histogram of Ad Selections",
     xlab = "Ad Index", ylab = "Number of Selections",
     breaks = d, border = "black")

# Print total reward
print(paste("Total Reward:", total_reward))
