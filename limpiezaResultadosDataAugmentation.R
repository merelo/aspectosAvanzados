dat<-read.xlsx("resultados.xlsx",sheetName = "Hoja1",header = FALSE)
dataAugm<-levels(dat$X1)
data<-levels(dat$X2)

ordenar<-function(x){
  x<-sapply(x,function(xx){unlist(strsplit(xx," "))})
  x<-t(x)
  rownames(x)<-NULL
  x<-matrix(c(as.numeric(x[,1]),as.numeric(x[,2])),ncol=2,byrow=FALSE)
  x<-x[order(x[,1]),]
  x
}

dataAugm<-ordenar(dataAugm)
data<-ordenar(data)

result<-data.frame(data)
result<-cbind(result,dataAugm[,2])
names(result)<-c("n_intentos","sinDA","conDA")

diferencia<-result$conDA-result$sinDA
resultado<-ifelse(diferencia>0,"mejora","empeora")
result<-cbind(result,diferencia,resultado)

table(resultado)
sum(diferencia)
#names(result)<-c(names(result),"diferencia")
