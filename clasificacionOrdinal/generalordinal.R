library(RWeka)
library(partykit)
datos<-read.arff("clasificacionOrdinal/esl.arff")
datos[,5]<-as.factor(datos[,5])

set.seed (2)
trainVal<-sample(1:nrow(datos),100)
test<-datos [trainVal ,]
train<-datos[-trainVal,]

clasifOrdinal<-function(train,test,modelo){
  clase<-length(train)
  nomClases<-levels(train[,clase])
  nClases<-length(nomClases)
  datos<-data.frame(matrix(nrow=nrow(test),ncol=0))
  trainAux<-train
  probabilidad<-data.frame(matrix(nrow=nrow(test),ncol=0))
  for(i in 1:(nClases-1)){
    cero<-which(trainAux[clase]==0)
    trainAux[clase]<-factor(ifelse(train[,clase]==nomClases[i],0,1))
    trainAux[cero,clase]<-0
    modelC4.5<-J48("out1~.", data=trainAux)
    predi<-predict(modelC4.5,test,type="probability")
    datos<-cbind(datos,predi[,1])
    datos<-cbind(datos,predi[,2])
    if(i==1){
      probabilidad<-cbind(probabilidad,datos[,1])
    }else{
      probabilidad<-cbind(probabilidad,(datos[,((i-1)*2)]*datos[,(i*2-1)]))
    }
  }
  probabilidad<-cbind(probabilidad,datos[,(nClases-1)*2])
  apply(probabilidad,1,function(x,nom){nom[which(x==max(x))]},nomClases)
}
datt<-clasifOrdinal(train,test,out1~.)

#prediccion frente a clase real
table(datt,test$out1)
#% acierto en la clasificacion
print(paste("Acierto: ",sum(datt==test$out1)/length(datt)*100,"%",sep=""))
