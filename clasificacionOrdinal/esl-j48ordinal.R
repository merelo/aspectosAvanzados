library(RWeka)
library(partykit)


datos<-read.arff("clasificacionOrdinal/esl.arff")
datos[,5]<-as.factor(datos[,5])

set.seed (2)
train<-sample(1:nrow(datos),100)
test<-datos [train ,]


modelC4.5 = J48(out1~., data=datos, subset=train)


plot(modelC4.5)
modelC4.5.pred = predict(modelC4.5, test,"probability")

modelC4.5.pred
