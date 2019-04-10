# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv' )
dataset = dataset[2:3]

# Fitting the Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level5 = dataset$Level^5
dataset$Level6 = dataset$Level^6
poly_reg = lm(formula = Salary ~ ., data = dataset)


# predict a single point
y_pred = predict(
  poly_reg,
  data.frame(
    Level = 6.5,
    Level2 = 6.5^2,
    Level3 = 6.5^3,
    Level4 = 6.5^4,
    Level5 = 6.5^5,
    Level6 = 6.5^6
  )
  )

# visualizing the results -- install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level, y = dataset$Salary), colour = 'red')+
  geom_line(aes(x=dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'blue')



