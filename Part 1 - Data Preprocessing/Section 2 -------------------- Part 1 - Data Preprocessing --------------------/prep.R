
dataset = read.csv("Data.csv")

# take care of missing data
dataset$Age = ifelse(
  is.na(dataset$Age),
  ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
  dataset$Age
)

dataset$Salary = ifelse(
  is.na(dataset$Salary),
  ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
  dataset$Salary
)

# enconde categorical data
dataset$Country = factor(
  dataset$Country, levels = c('France', 'Spain', 'Germany'), labels = c(1, 2, 3)
) # this labels are non-numeric but labels, even if they contain numbers

dataset$Purchased = factor(
  dataset$Purchased, levels = c('No', 'Yes'), labels = c(0, 1)
)


# split the dataset into training and testing sets
# install.packages('catools')
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) # SplitRatio is the percentaje of the dataset that goes to the training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# scale the training sets
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3]) # we are using 2:3 to exclude column 1, which is non-numeric

















