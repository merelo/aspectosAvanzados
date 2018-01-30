#circle

# imbalanced.R
# Implementation and evaluation of imbalanced classification techniques 
# Programming code courtesy by Sarah Vluymans, Sarah.Vluymans@UGent.be

## load the circle dataset
circle <- read.table("clasifNoBalanceado/circle.txt", sep=",")
colnames(circle) <- c("Att1", "Att2", "Class")

# determine the imbalance ratio
unique(circle$Class)
nClass0 <- sum(circle$Class == 0)
nClass1 <- sum(circle$Class == 1)
IR <- nClass1 / nClass0
IR #aproximacion de la dificultad del problema

# visualize the data distribution
plot(circle$Att1, circle$Att2)
points(circle[circle$Class==0,1],circle[circle$Class==0,2],col="red")
points(circle[circle$Class==1,1],circle[circle$Class==1,2],col="blue")  

# Set up the dataset for 5 fold cross validation. 5 suele ser mejor
# Make sure to respect the class imbalance in the folds.
pos <- (1:dim(circle)[1])[circle$Class==0]
neg <- (1:dim(circle)[1])[circle$Class==1]

CVperm_pos <- matrix(sample(pos,length(pos)), ncol=5, byrow=T)
CVperm_neg <- matrix(sample(neg,length(neg)), ncol=5, byrow=T)

CVperm <- rbind(CVperm_pos, CVperm_neg)

# Base performance of 3NN
library(class)
knn.pred = NULL
for( i in 1:5){
  predictions <- knn(circle[-CVperm[,i], -3], circle[CVperm[,i], -3], circle[-CVperm[,i], 3], k = 3)
  knn.pred <- c(knn.pred, predictions)
}
acc <- sum((circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) 
           | (circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2)) / (nClass0 + nClass1)
tpr <- sum(circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr <- sum(circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean <- sqrt(tpr * tnr)


# 1. ROS random oversampling
knn.pred = NULL
for( i in 1:5){
  
  train <- circle[-CVperm[,i], -3]
  classes.train <- circle[-CVperm[,i], 3] 
  test  <- circle[CVperm[,i], -3]
  
  # randomly oversample the minority class (class 0)
  minority.indices <- (1:dim(train)[1])[classes.train == 0]
  to.add <- dim(train)[1] - 2 * length(minority.indices) #por 2, duplica
  duplicate <- sample(minority.indices, to.add, replace = T)
  for( j in 1:length(duplicate)){
    train <- rbind(train, train[duplicate[j],]) #duplicar
    classes.train <- c(classes.train, 0)
  }  
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.ROS <- sum(circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.ROS <- sum(circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.ROS <- sqrt(tpr.ROS * tnr.ROS)

# 2. RUS random undersampling
knn.pred = NULL
for( i in 1:5){
  
  train <- circle[-CVperm[,i], -3]
  classes.train <- circle[-CVperm[,i], 3] 
  test  <- circle[CVperm[,i], -3]
  
  # randomly undersample the minority class (class 1)
  majority.indices <- (1:dim(train)[1])[classes.train == 1]
  to.remove <- 2* length(majority.indices) - dim(train)[1]
  remove <- sample(majority.indices, to.remove, replace = F)
  train <- train[-remove,] 
  classes.train <- classes.train[-remove]
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.RUS <- sum(circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.RUS <- sum(circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.RUS <- sqrt(tpr.RUS * tnr.RUS)

# Visualization (RUS on the full dataset)
circle.RUS <- circle
majority.indices <- (1:dim(circle.RUS)[1])[circle.RUS$Class == 1]
to.remove <- 2 * length(majority.indices) - dim(circle.RUS)[1]
remove <- sample(majority.indices, to.remove, replace = F)
circle.RUS <- circle.RUS[-remove,] 

plot(circle.RUS$Att1, circle.RUS$Att2)
points(circle.RUS[circle.RUS$Class==0,1],circle.RUS[circle.RUS$Class==0,2],col="red")
points(circle.RUS[circle.RUS$Class==1,1],circle.RUS[circle.RUS$Class==1,2],col="blue") 
