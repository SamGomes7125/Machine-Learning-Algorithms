# Eclat Algorithm for Association Rule Learning

# Install required packages if not already installed
if (!require(arules)) install.packages("arules", dependencies=TRUE)

# Load necessary library
library(arules)

# Load the dataset as transactions
dataset <- read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)

# Dataset Summary
summary(dataset)

# Visualizing the top 10 most frequent items
itemFrequencyPlot(dataset, topN = 10, col = "steelblue", main = "Top 10 Most Frequent Items")

# Applying Eclat Algorithm
# Support = (3 purchases per day * 7 days) / Total transactions (7501)
rules <- eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Sorting & Inspecting the top 10 frequent itemsets by support
inspect(sort(rules, by = 'support')[1:10])
