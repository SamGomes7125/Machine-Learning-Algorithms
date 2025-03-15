# Natural Language Processing (NLP) - Sentiment Analysis

# Importing Dataset
dataset_original <- read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Loading necessary libraries
library(tm)
library(SnowballC)
library(caTools)
library(e1071)

# Cleaning the texts
corpus <- VCorpus(VectorSource(dataset_original$Review))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("en"))  # Explicitly setting language
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.997)  # Adjusted for better feature selection

# Convert DTM to dataframe
dataset <- as.data.frame(as.matrix(dtm))
dataset$liked <- factor(dataset_original$Liked)  # Ensure factor conversion

# Splitting dataset into Training and Test sets
set.seed(123)
split <- sample.split(dataset$liked, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Training Naive Bayes Classifier
classifier <- naiveBayes(x = training_set[, -ncol(training_set)],  # Exclude target column
                         y = training_set$liked)

# Predicting Test set results
y_pred <- predict(classifier, newdata = test_set[, -ncol(test_set)])  # Ensure proper column indexing

# Making Confusion Matrix
cm <- table(test_set$liked, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculating Accuracy
accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))
