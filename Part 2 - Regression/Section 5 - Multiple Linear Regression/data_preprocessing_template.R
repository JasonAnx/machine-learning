# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding Categorical Data
dataset$State = factor(
  dataset$State, 
  levels = c('New York', 'California', 'Florida'),
  labels = c(1, 2, 3)
)

# Splitting the dataset into the Training set and Test set
# ---install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling --- not needed for linear regression on R
# training_set = scale(training_set)
# test_set = scale(test_set)


# fitting Multiple Linear Regression to the training set
regressor = lm(
  formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
  data = training_set
) # -- read as: profit is a linear combination of ~
# ---- you could also use Profit ~ .
# ---- the dot means all the other variables
# ---- 
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = training_set ) # redefined, exluding high p-val cols
y_pred = predict(regressor, newdata = test_set)











