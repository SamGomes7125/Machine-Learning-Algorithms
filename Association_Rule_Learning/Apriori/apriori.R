# Apriori Algorithm for Association Rule Learning

# Install required packages if not already installed
if (!require(arules)) install.packages("arules", dependencies=TRUE)
if (!require(arulesViz)) install.packages("arulesViz", dependencies=TRUE)

# Load necessary libraries
library(arules)
library(arulesViz)

# Load the dataset as transactions
dataset <- read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)

# Dataset Summary
summary(dataset)

# Visualizing the top 10 most frequent items
itemFrequencyPlot(dataset, topN = 10, col = "steelblue", main = "Top 10 Most Frequent Items")

# Applying Apriori Algorithm
# Support = (3 purchases per day * 7 days) / Total transactions (7501)
rules <- apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2, minlen = 2))

# Sorting & Inspecting the top 10 rules by lift
inspect(sort(rules, by = 'lift')[1:10])

# Visualizing Association Rules
plot(rules, method = "graph", engine = "htmlwidget")
