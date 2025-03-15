# Thompson Sampling Algorithm in R

# Importing the dataset
dataset <- read.csv('Ads_CTR_Optimisation.csv')

# Parameters
N <- 10000  # Number of rounds
d <- 10     # Number of ads
ads_selected <- integer(0)
numbers_of_rewards_1 <- rep(0, d)  # Count of rewards = 1
numbers_of_rewards_0 <- rep(0, d)  # Count of rewards = 0
total_reward <- 0

# Thompson Sampling Algorithm
for (n in 1:N) {
  max_random <- 0
  ad <- 1  # Index starts from 1 in R
  
  for (i in 1:d) {
    random_beta <- rbeta(1, numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
    if (random_beta > max_random) {
      max_random <- random_beta
      ad <- i
    }
  }
  
  # Select ad and get reward
  ads_selected <- c(ads_selected, ad)
  reward <- dataset[n, ad]
  
  if (reward == 1) {
    numbers_of_rewards_1[ad] <- numbers_of_rewards_1[ad] + 1
  } else {
    numbers_of_rewards_0[ad] <- numbers_of_rewards_0[ad] + 1
  }
  
  total_reward <- total_reward + reward
}

# Visualizing the Results
hist(ads_selected, col = 'blue', main = "Histogram of Ads Selection",
     xlab = "Ad Index", ylab = "Number of Selections",
     breaks = d, border = "black")

# Print total reward
print(paste("Total Reward:", total_reward))
