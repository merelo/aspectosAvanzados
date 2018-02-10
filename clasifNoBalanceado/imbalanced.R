# imbalanced.R
# Implementation and evaluation of imbalanced classification techniques 
# Programming code courtesy by Sarah Vluymans, Sarah.Vluymans@UGent.be

## load the subclus dataset
subclus <- read.table("clasifNoBalanceado/subclus.txt", sep=",")
colnames(subclus) <- c("Att1", "Att2", "Class")

# determine the imbalance ratio
unique(subclus$Class)
nClass0 <- sum(subclus$Class == 0)
nClass1 <- sum(subclus$Class == 1)
IR <- nClass1 / nClass0
IR #aproximacion de la dificultad del problema

# visualize the data distribution
plot(subclus$Att1, subclus$Att2)
points(subclus[subclus$Class==0,1],subclus[subclus$Class==0,2],col="red")
points(subclus[subclus$Class==1,1],subclus[subclus$Class==1,2],col="blue")  

# Set up the dataset for 5 fold cross validation. 5 suele ser mejor
# Make sure to respect the class imbalance in the folds.
pos <- (1:dim(subclus)[1])[subclus$Class==0]
neg <- (1:dim(subclus)[1])[subclus$Class==1]

CVperm_pos <- matrix(sample(pos,length(pos)), ncol=5, byrow=T)
CVperm_neg <- matrix(sample(neg,length(neg)), ncol=5, byrow=T)

CVperm <- rbind(CVperm_pos, CVperm_neg)

# Base performance of 3NN
library(class)
knn.pred = NULL
for( i in 1:5){
  predictions <- knn(subclus[-CVperm[,i], -3], subclus[CVperm[,i], -3], subclus[-CVperm[,i], 3], k = 3)
  knn.pred <- c(knn.pred, predictions)
}
acc <- sum((subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) 
           | (subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2)) / (nClass0 + nClass1)
tpr <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean <- sqrt(tpr * tnr)


# 1. ROS random oversampling
knn.pred = NULL
for( i in 1:5){
  
  train <- subclus[-CVperm[,i], -3]
  classes.train <- subclus[-CVperm[,i], 3] 
  test  <- subclus[CVperm[,i], -3]
  
  # randomly oversample the minority class (class 0)
  minority.indices <- (1:dim(train)[1])[classes.train == 0]
  to.add <- dim(train)[1] - 2 * length(minority.indices)
  duplicate <- sample(minority.indices, to.add, replace = T)
  for( j in 1:length(duplicate)){
    train <- rbind(train, train[duplicate[j],]) #duplicar
    classes.train <- c(classes.train, 0)
  }  
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.ROS <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.ROS <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.ROS <- sqrt(tpr.ROS * tnr.ROS)

# 2. RUS random undersampling
knn.pred = NULL
for( i in 1:5){
  
  train <- subclus[-CVperm[,i], -3]
  classes.train <- subclus[-CVperm[,i], 3] 
  test  <- subclus[CVperm[,i], -3]
  
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
tpr.RUS <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.RUS <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.RUS <- sqrt(tpr.RUS * tnr.RUS)

# Visualization (RUS on the full dataset)
subclus.RUS <- subclus
majority.indices <- (1:dim(subclus.RUS)[1])[subclus.RUS$Class == 1]
to.remove <- 2 * length(majority.indices) - dim(subclus.RUS)[1]
remove <- sample(majority.indices, to.remove, replace = F)
subclus.RUS <- subclus.RUS[-remove,] 

plot(subclus.RUS$Att1, subclus.RUS$Att2)
points(subclus.RUS[subclus.RUS$Class==0,1],subclus.RUS[subclus.RUS$Class==0,2],col="red")
points(subclus.RUS[subclus.RUS$Class==1,1],subclus.RUS[subclus.RUS$Class==1,2],col="blue") 


# 1.4.1 Distance function
distance <- function(i, j, data){
  sum <- 0
  for(f in 1:dim(data)[2]){
    if(is.factor(data[,f])){ # nominal feature
      if(data[i,f] != data[j,f]){
        sum <- sum + 1
      }
    } else {
      sum <- sum + (data[i,f] - data[j,f]) * (data[i,f] - data[j,f])
    }
  }
  sum <- sqrt(sum)
  return(sum)
}


#4.4.2 getNeighbors
getNeighbors<-function(x,minority.instances,train){
  respuesta<-matrix(ncol=2,nrow=0)
  for(i in minority.instances){
    if(x!=i){
      respuesta<-rbind(respuesta,c(i,distance(x,i,train)))
    }
  }
  return(head(respuesta[order(respuesta[,2]),1],n=5))
}

#3.4.3 syntheticInstance
syntheticInstance <- function(x, neighbors, data){
  y<-neighbors[sample(1:length(neighbors),1)]
  porcentaje<-runif(1,min=0,max=1)
  
  xx<-(abs(abs(data[x,1])-abs(data[y,1]))*porcentaje)+min(c(data[x,1],data[y,1]))
  resp<-sapply(2:ncol(data),function(col,x,y,data,porcentaje,xx){
    if(class(data[x,col])!="factor"){
      pend<-(data[x,col]-data[y,col])/(data[x,1]-data[y,1])
      b<-data[x,col]-data[x,1]*pend
      ifelse(data[x,1]==data[y,1],data[y,1],pend*xx+b)
    }else
      ifelse(porcentaje<0.5,data[x,col],data[y,col])
      #sample(c(data[x,col],data[y,col]),1)
    c(data[x,col],data[y,col],data[x,1],data[y,1])
  },x,y,data,porcentaje,xx)
  c(xx,resp)
}

syntheticInstance(1,getNeighbors(1,c(2,3,4,5),train),train)


# SMOTE
knn.pred = NULL
for( i in 1:5){
  
  train <- subclus[-CVperm[,i], -3]
  classes.train <- subclus[-CVperm[,i], 3] 
  test  <- subclus[CVperm[,i], -3]
  
  # randomly oversample the minority class (class 0)
  minority.indices <- (1:dim(train)[1])[classes.train == 0]
  to.add <- dim(train)[1] - 2 * length(minority.indices)
  duplicate <- sample(minority.indices, to.add, replace = T)
  for( j in 1:length(duplicate)){
    train <- rbind(train, syntheticInstance(duplicate[j],getNeighbors(duplicate[j],minority.indices,train),train))
    classes.train <- c(classes.train, 0)
  }  
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.SMOTE <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.SMOTE <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.SMOTE <- sqrt(tpr.SMOTE * tnr.SMOTE)

gmean.ROS
gmean.RUS
gmean.SMOTE #comparando los tres metodos, vemos que SMOTE mejora los otros dos en nuestro caso

# Visualization (SMOTE on the full dataset)
classes.train <- subclus[, 3] 
subclus.SMOTE <- subclus
minority.indices <- (1:dim(subclus.SMOTE)[1])[classes.train == 0]
trainTot<-subclus[-3]
to.add <- dim(trainTot)[1] - 2 * length(minority.indices)
duplicate <- sample(minority.indices, to.add, replace = T)
classes.trainTot<-c()

# subclus[3,] #290 -57
# subclus[14,] #288 -84
# subclus[8,] #289 4
# subclus[17,] #
# subclus[16,] #288 -84
# subclus[6,] #288 -84
# plot(matrix(c(196,-81,258,1,280,6,289,4,290,-57,288,-84,syntheticInstance(3,getNeighbors(3,minority.indices,trainTot),trainTot)),ncol=2,byrow=TRUE),col=c("blue","blue","blue","blue","black","blue","red"))


for( j in 1:length(duplicate)){
  trainTot <- rbind(trainTot, syntheticInstance(duplicate[j],getNeighbors(duplicate[j],minority.indices,trainTot),trainTot))
  classes.trainTot <- c(classes.trainTot, 2) #le atribuimos una clase nueva simplemente para que se vean los nuevos valores imputados a la clase 0
}  
subclus.SMOTE <- cbind(trainTot,c(subclus$Class,classes.trainTot))
names(subclus.SMOTE)<-c("Att1","Att2","Class")

plot(subclus.SMOTE$Att1, subclus.SMOTE$Att2)
points(subclus.SMOTE[subclus.SMOTE$Class==0,1],subclus.SMOTE[subclus.SMOTE$Class==0,2],col="red")
points(subclus.SMOTE[subclus.SMOTE$Class==1,1],subclus.SMOTE[subclus.SMOTE$Class==1,2],col="blue") 
points(subclus.SMOTE[subclus.SMOTE$Class==2,1],subclus.SMOTE[subclus.SMOTE$Class==2,2],col="green") #imputaciones con SMOTE de la clase 0
