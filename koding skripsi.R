# Set working directory

setwd("D:/SKRIPSI ANDRES")

#	KESELURUHAN KONTEN 	
# read in the data install.packages ("readr") library(readr)

analisis1 <- read.csv("data_cleaning2.csv",header = TRUE)

##############################(LANGKAH 1)#######################################
# Script R Preprocessing Data dengan Text Mining
# Install

install.packages("tm") # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud")
install.packages("wordcloud2")# word-cloud generator 
install.packages("RColorBrewer") # color palettes
install.packages("stringr")
install.packages("plyr")
install.packages("RTextTools")

# Load

library("tm")
library("SnowballC")
library("wordcloud")
library("wordcloud2")
library("RColorBrewer")
library("stringr")
library("plyr")

setwd("D:/SKRIPSI ANDRES")
docs<-readLines("data_cleaning2.csv")

# Load the data as a corpus

docs <- Corpus(VectorSource(docs))

#Inspect the content of the document

inspect(docs)

#Replacing ?/?, ?@? and ?|? with space:

toSpace <- content_transformer(function (x , pattern ) 
gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")

#Cleaning the text
#Remove punctuation

docs <- tm_map(docs, toSpace, "[[:punct:]]")

#Remove numbers

docs <- tm_map(docs, toSpace, "[[:digit:]]")

# Convert the text to lower case
docs <- tm_map(docs, content_transformer (tolower))

# add two extra stop words: "available" and "via"

myStopwords = readLines("stopwordsid.csv")

# remove stopwords from corpus

docs <- tm_map(docs, removeWords, myStopwords)

# Remove your own stop word
# specify your stopwords as a character vector

docs <- tm_map(docs, removeWords,  c("dan", "yang", "untuk", "dari","lagi", "kami",
"jika", "pun", "selain", "tersebut", "serta","karena", "bersama", "pada", "itu",
"dalam", "di","juga", "melalui", "menjadi", "melakukan", "adalah", "jadi", "saja", 
"ingin", "selengkapnya", "tentang", "hal", "bisa", "juga", "tetap", "tapi",
"hanya", "yaitu", "bagi", "seperti", "suatu", "ini", "dapat", "sebagai", "harus", 
"sesuai","dengan", "akan", "per", "ada", "telah",  "para", "masa", "tidak", 
"secara", "atau", "tak", "oleh", "kita","apa", "agar", "sudah","kepada", "saat",
"yukks","wkwk", "bgt","min","tpi", "weh", "loh", "tik","klo", "juga","nah", "aja",
"jdi","iya", "dpt", "jok", "nya", "kah","for","pst","ntr"))

# Eliminate extra white spaces

docs <- tm_map(docs, stripWhitespace)

# Remove URL

removeURL <- function(x) gsub("http[[:alnum:]]*", " ", x)
docs <- tm_map(docs, removeURL)

#Build a term-document matrix

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
dataframe<-data.frame(text=unlist(sapply(docs, `[`)), 
                      stringsAsFactors=F)

write.csv(dataframe, "D:/SKRIPSI ANDRES/data_cleaning3.csv")
save.image()

#WORDCLOUD
# Deklarasi 

corpus <- Corpus(VectorSource(dataframe))
corpus [1][1]
View(corpus)

tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
v <- sort(rowSums(m), decreasing = TRUE)
d <- data.frame(word =names(v), freq=v)

# Plotting Word cloud

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 0.5,
          max.words=3000, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

###############################(Langkah 2)##################################### 
# Script R Pelabelan dan Pembobotan

kalimat2<-read.csv("data_cleaning3.csv",header=TRUE)

#skoring data berdasarkan kamus kata

positif <- scan("positif.txt",what="character",comment.char=";")
negatif <- scan("negatif.txt",what="character",comment.char=";")
kata.positif = c(positif)
kata.negatif = c(negatif)
score.sentiment = function(kalimat2, kata.positif, kata.negatif,
                           .progress='none')

  {
    require(plyr)
    require(stringr)
    scores = laply(kalimat2, function(kalimat, kata.positif, kata.negatif)
    {
      kalimat = gsub('[[:punct:]]', '', kalimat)
      kalimat = gsub('[[:cntrl:]]', '', kalimat)
      kalimat = gsub('\\d+', '', kalimat)
      kalimat = tolower(kalimat)
      
      list.kata = str_split(kalimat, '\\s+')
      kata2 = unlist(list.kata)
      positif.matches = match(kata2, kata.positif)
      negatif.matches = match(kata2, kata.negatif)
      positif.matches = !is.na(positif.matches)
      negatif.matches = !is.na(negatif.matches)
      score = sum(positif.matches) - (sum(negatif.matches))
      return(score)
    }, kata.positif, kata.negatif, .progress=.progress )
    scores.df = data.frame(score=scores, text=kalimat2)
    return(scores.df)
  }

hasil = score.sentiment(kalimat2$text, kata.positif, kata.negatif)
head (hasil)
View(hasil)

#CONVERT SCORE TO SENTIMENT 
hasil$klasifikasi <- ifelse(hasil$score<0, "Negatif", "Positif")
hasil$klasifikasi
View(hasil)

#Tukar Row

data <- hasil[c(3,1,2)]
View(data)
write.csv(data, "pelabelan.csv")

#Menyimpan data positif dan negatif

data.pos <- hasil[hasil$score>0,]
View(data.pos)
write.csv(data.pos, "test_positif.csv")

data.neg <- hasil[hasil$score<0,]
View(data.neg)
write.csv(data.neg, "test_negatif.csv")

###############################################################################
##################################(Langkah 3)##################################
#Script R Klasifikasi dengan menggunakan metode Naïve Bayes Classifier
#Preprocessing
#CARA 1

setwd("D:/SKRIPSI ANDRES")

install.packages("dplyr") # grammar data manipulation
install.packages("e1071") #Data Classification
install.packages("caret") # Classification And REgression Training
install.packages("devtools")
install.packages("MIAmaxent")
install.packages("RTextTools")
install.packages("tm")
install.packages("doMC", repos="http://R-Forge.R-project.org")
install.packages("readr")

# Load required libraries

library("tm")
library("RTextTools")
library("e1071")
library("dplyr")
library("caret")
library("SparseM")
library("RTextTools")
library("e1071")
library("stats")
library("dplyr")
library("NLP")
library("caret")
library("MIAmaxent")
library("readr")

# Library for parallel processing
library("doMC")
registerDoMC(cores = detectCores())

#input data

library(readr)
df <- read.csv("pelabelan.csv", stringsAsFactors=FALSE, encoding="UTF8")
View(df)
glimpse(df) 

# Random dataset
set.seed(1)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
glimpse(df) 

#tokenization

df$Kalimat<- as.factor(df$Kalimat)
corpus <- Corpus(VectorSource(df$Kalimat))
corpus
inspect(corpus[1:3])

#Clean up
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus)
inspect(dtm[40:50, 10:15]) 

#PEMBAGIAN DATA LATIH 60% DAN DATA UJI 40%

df.train <- df[1:1106,]
df.test <- df[1107:1843,]
dtm.train <- dtm[1:1106,]
dtm.test <- dtm[1107:1843,]
corpus.train <- corpus[1:1106]
corpus.test <- corpus[1107:1843]

dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 10)
length((fivefreq))
fivefreq
dtm.train.nb <- DocumentTermMatrix(corpus.train, control=list(dictionary= fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.test, control=list(dictionary = fivefreq))
dim(dtm.test.nb) 

#Fungsi Pelabelan

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

library(naivebayes)
naive=system.time( classifier <- naiveBayes(trainNB, df.train$Kalimat,laplace = 0) )
print(naive)
system.time( pred <- predict(classifier, newdata=testNB) )

# Confusion Matrix

conf.mat <- confusionMatrix(table("Predictions"= pred, "Actual" = df.test$Kalimat ))
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy'] 

#PEMBAGIAN DATA LATIH 70% DAN DATA UJI 30%

df.train <- df[1:1290,]
df.test <- df[1291:1843,]
dtm.train <- dtm[1:1290,]
dtm.test <- dtm[1291:1843,]
corpus.train <- corpus[1:1290]
corpus.test <- corpus[1291:1843] 

dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 10)
length((fivefreq))
fivefreq
dtm.train.nb <- DocumentTermMatrix(corpus.train, control=list(dictionary= fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.test, control=list(dictionary = fivefreq))
dim(dtm.test.nb) 

#Fungsi Pelabelan

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

library(naivebayes)
naive=system.time( classifier <- naiveBayes(trainNB, df.train$Kalimat,laplace = 0) )
print(naive)
system.time( pred <- predict(classifier, newdata=testNB) )

# Confusion Matrix
conf.mat <- confusionMatrix(table("Predictions"= pred, "Actual" = df.test$Kalimat ))
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy'] 

#PEMBAGIAN DATA LATIH 80% DAN DATA UJI 20%

df.train <- df[1:1474,]
df.test <- df[1475:1843,]
dtm.train <- dtm[1:1474,]
dtm.test <- dtm[1475:1843,]
corpus.train <- corpus[1:1474]
corpus.test <- corpus[1475:1843] 

dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 10)
length((fivefreq))
fivefreq
dtm.train.nb <- DocumentTermMatrix(corpus.train, control=list(dictionary= fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.test, control=list(dictionary = fivefreq))
dim(dtm.test.nb) 

#Fungsi Pelabelan

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

library(naivebayes)
naive=system.time( classifier <- naiveBayes(trainNB, df.train$Kalimat,laplace = 0) )
print(naive)
system.time( pred <- predict(classifier, newdata=testNB) )

# Confusion Matrix
conf.mat <- confusionMatrix(table("Predictions"= pred, "Actual" = df.test$Kalimat ))
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy'] 

#PEMBAGIAN DATA LATIH 90% DAN DATA UJI 10%

df.train <- df[1:1567,]
df.test <- df[1568:2733,]
dtm.train <- dtm[1:1567,]
dtm.test <- dtm[1568:2733,]
corpus.train <- corpus[1:1567]
corpus.test <- corpus[1568:2733] 

dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 10)
length((fivefreq))
fivefreq
dtm.train.nb <- DocumentTermMatrix(corpus.train, control=list(dictionary= fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.test, control=list(dictionary = fivefreq))
dim(dtm.test.nb) 


#Fungsi Pelabelan

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

install.packages("naivebayes")
library(naivebayes)
naive=system.time( classifier <- naiveBayes(trainNB, df.train$Kalimat,laplace = 0) )
print(naive)
system.time( pred <- predict(classifier, newdata=testNB) )

# Confusion Matrix
conf.mat <- confusionMatrix(table("Predictions"= pred, "Actual" = df.test$Kalimat ))
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy'] 


###############################################################################
#CARA 2

setwd("D:/SKRIPSI ANDRES")

#Input Data

df<- read.csv("pelabelan.csv", stringsAsFactors = TRUE)
glimpse(df)
View(df)
write.csv(df,file = "df10.csv")

#Randomize Dataset

set.seed(10)
df <- df[sample(nrow(df)),]
df <- df[sample(nrow(df)),]
glimpse(df)
write.csv(df,file = "df10.csv")

#Tokenization

corpus <- Corpus(VectorSource(df$kalimat))
corpus
inspect(corpus[1:5])

#Clean up
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(removeWords, c("mengapa","apasih","dan","please","yg","dari",
"lg","oleh","dalam","uwow","ini","untuk","sekarang","dulu","aja","jos",
"brengsek","siapa","jd","hehe","kenapasih","ayo","lagi","tapi","karena",
"saat","ada","sudah","atas","njir","ga","aing","soal","jd","lo",
"minta","please","ngga","masih","semua")) %>%
tm_map(stripWhitespace)

#Matriks

dtm <- DocumentTermMatrix(corpus.clean)
inspect(dtm [40:50, 10:15])

#Partitioning

df.train <- df[1:3072,]
df.test <- df[3073:3840,]
dtm.train <- dtm[1:3072,]
dtm.test <- dtm[2073:2840,]
corpus.clean.train <- corpus.clean[1:2072]
corpus.clean.test <- corpus.clean[2073:2840]
write.csv(df.train,file = "TrainNB10.csv")
write.csv(df.test,file = "TestNB10.csv")

#Featured Selection

dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train,1)
length((fivefreq))
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train,
                                   control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test,
                                  control=list(dictionary = fivefreq))
dim(dtm.train.nb)

#Boolan Naive Bayes
convert_countNB <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
  
}

#NAIVE BAYES CLSSIFIER

df=read.csv2("df10.csv")
#Naive Bayes
TrainNB=read.csv2("TrainNB10.csv")
TestNB=read.csv2("TestNB10.csv")
trainNB <- apply(dtm.train.nb, 2, convert_countNB)
testNB <- apply(dtm.test.nb, 2, convert_countNB)

#Training
classifier <- naiveBayes(trainNB, df.train$class, laplace = 1)
#Testing
pred <- predict(classifier, testNB)
#Tabel Prediksi vs Aasli
table("Predictions"= pred, "Actual" = df.test$class )

##confusion matriks
# Prepare the confusion matrix
conf.mat <- confusionMatrix(pred, df.test$class)
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy']*100

################################################################################

####################################(LANGKAH 4)################################
#Script R Visualisasi dan Asosiasi Teks (Untuk Ulasan Positif)
# Install

install.packages("tm") # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator
install.packages("RColorBrewer") # color palettes

# Load

library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library("stringr")

setwd("D:/SKRIPSI ANDRES")
# docs<- readLines("positif.csv")

docs<- readLines("positiv.csv")

# Load the data as a corpus

docs <- Corpus(VectorSource(docs))

# Remove your own stop word
# specify your stopwords as a character vector

docs <- tm_map(docs, removeWords, c("min","pst","mendapatkan","sekarang",
"syarat","tinggi","kali","full","bang","afc","doll","status", "udh","lbih",
"kurang","cil","jdi","nih","sih","yuk","son","ini","bet", "xfd","ada","sedikit",
"dalam","terdapat","aja","semua","dengan","bener","tks","tidak","noror","ball",
"kerena","diatas","super","pemula","lg","bisnis","download","min","buat",
"diam","terakhir","can","tuh","bet","santai","aja","yg", "enggak", "ilu",
"apaik", "adaail"))

# Eliminate extra white spaces

docs <- tm_map(docs, stripWhitespace)

#Replace words

docs <- tm_map(docs, gsub, pattern="banayk",replacement="banyak")
docs <- tm_map(docs, gsub, pattern="yg",replacement="yang")
docs <- tm_map(docs, gsub, pattern="bsa",replacement="bisa")
docs <- tm_map(docs, gsub, pattern="bnyk",replacement="banyak")
docs <- tm_map(docs, gsub, pattern="bs",replacement="bisa")
docs <- tm_map(docs, gsub, pattern="mntp",replacement="mantap")
docs <- tm_map(docs, gsub, pattern="jd",replacement="jadi")
docs <- tm_map(docs, gsub, pattern="tidk",replacement="tidak")
docs <- tm_map(docs, gsub, pattern="tdak",replacement="tidak")
docs <- tm_map(docs, gsub, pattern="gmn",replacement="gimana")
docs <- tm_map(docs, gsub, pattern="bg",replacement="bang")
docs <- tm_map(docs, gsub, pattern="jngn",replacement="jangan")
docs <- tm_map(docs, gsub, pattern="kmn",replacement="kemana")
docs <- tm_map(docs, gsub, pattern="kmna",replacement="kemana")
docs <- tm_map(docs, gsub, pattern="lg",replacement="lagi")
docs <- tm_map(docs, gsub, pattern="udh",replacement="sudah")
docs <- tm_map(docs, gsub, pattern="knp",replacement="kenapa")
docs <- tm_map(docs, gsub, pattern="blm",replacement="belom")
docs <- tm_map(docs, gsub, pattern="ig",replacement="instagram")
docs <- tm_map(docs, gsub, pattern="mlah",replacement="malah")
docs <- tm_map(docs, gsub, pattern="ap",replacement="apa")
docs <- tm_map(docs, gsub, pattern="ad",replacement="ada")
docs <- tm_map(docs, gsub, pattern="lgi",replacement="lagi")
docs <- tm_map(docs, gsub, pattern="maen",replacement="main")
docs <- tm_map(docs, gsub, pattern="jg",replacement="juga")
docs <- tm_map(docs, gsub, pattern="knpa",replacement="kenapa")
docs <- tm_map(docs, gsub, pattern="aj",replacement="aja")
docs <- tm_map(docs, gsub, pattern="dri",replacement="dari")
docs <- tm_map(docs, gsub, pattern="dr",replacement="dari")
docs <- tm_map(docs, gsub, pattern="bs",replacement="bisa")
docs <- tm_map(docs, gsub, pattern="dlu",replacement="dulu")
docs <- tm_map(docs, gsub, pattern="dl",replacement="dulu")
docs <- tm_map(docs, gsub, pattern="kalo",replacement="kalau")
docs <- tm_map(docs, gsub, pattern="smngt",replacement="semangat")
docs <- tm_map(docs, gsub, pattern="lbh",replacement="lebih")
docs <- tm_map(docs, gsub, pattern="kyak",replacement="kayak")
docs <- tm_map(docs, gsub, pattern="kyk",replacement="kayak")
docs <- tm_map(docs, gsub, pattern="td",replacement="tadi")
docs <- tm_map(docs, gsub, pattern="skrng",replacement="sekarang")
docs <- tm_map(docs, gsub, pattern="ngk",replacement="enggak")
docs <- tm_map(docs, gsub, pattern="ttp",replacement="tetap")
docs <- tm_map(docs, gsub, pattern="bgt",replacement="banget")
docs <- tm_map(docs, gsub, pattern="bngt",replacement="banget")
docs <- tm_map(docs, gsub, pattern="menangg",replacement="menang")
docs <- tm_map(docs, gsub, pattern="dpn",replacement="depan")
docs <- tm_map(docs, gsub, pattern="klh",replacement="kalah")
docs <- tm_map(docs, gsub, pattern="skrg",replacement="sekarang")
docs <- tm_map(docs, gsub, pattern="brp",replacement="berapa")
docs <- tm_map(docs, gsub, pattern="krna",replacement="karena")
docs <- tm_map(docs, gsub, pattern="pnya",replacement="punya")
docs <- tm_map(docs, gsub, pattern="jls",replacement="jelas")
docs <- tm_map(docs, gsub, pattern="laen",replacement="lain")
docs <- tm_map(docs, gsub, pattern="makasi",replacement="terima kasih")
docs <- tm_map(docs, gsub, pattern="mks",replacement="terima kasih")
docs <- tm_map(docs, gsub, pattern="pake",replacement="pakai")
docs <- tm_map(docs, gsub, pattern="mna",replacement="mana")
docs <- tm_map(docs, gsub, pattern="bgtu",replacement="begitu")
docs <- tm_map(docs, gsub, pattern="dapaat",replacement="dapat")

#Build a term-document matrix

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 50)
dataframe<-data.frame(text=unlist(sapply(docs, `[`)), 
                      stringsAsFactors=F)


#Generate the Word cloud

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 0.5,
          max.words=1000, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

#
write.csv(dataframe, "D:/SKRIPSI ANDRES/data_positif.csv")
save.image()

#Explore frequent terms and their associations

findFreqTerms(dtm, lowfreq = 4)

#Find related words
v<-as.list(findAssocs(dtm, terms =c("persija","alhamdulillah","menang","semangat","bagus","baik")
,corlimit = c(0.15,0.15,0.15,0.15,0.15,0.15)))
v

#Find related words (one by one)

v<-as.list(findAssocs(dtm, terms =c("aplikasi"),
                      corlimit = c(0.15)))
View(v$aplikasi)

#barplot

k<-barplot(d[1:15,]$freq, las = 2, names.arg =
             d[1:15,]$word,cex.axis=1.2,cex.names=1.2,
           main ="Most frequent words",
           ylab = "Word frequencies",col = terrain.colors(20))
termFrequency <- rowSums(as.matrix(dtm))
termFrequency <- subset(termFrequency, termFrequency>=115)

text(k,sort(termFrequency, decreasing = T)-
       2,labels=sort(termFrequency, decreasing = T),pch = 6, cex = 1)

###############################################################################

#################################(LANGKAH 5)###################################
#Script R Visualisasi dan Asosiasi Teks (Untuk Ulasan Negatif)

docs<-readLines("negatif.csv")

# Load the data as a corpus

docs <- Corpus(VectorSource(docs))

# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
# Remove your own stop word
# specify your stopwords as a character vector

docs <- tm_map(docs, removeWords, c("min","pst","mendapatkan","sekarang","syarat",
"tinggi","kali","full","bang","afc","doll","status", "udh","lbih","jis","cil",
"jdi","nih","sih","yuk","son","ini","bet", "xfd","ada","sedikit","dalam",
"terdapat","aja","semua","dengan","bener","tks","tidak","noror","ball","kerena",
"diatas","super","pemula","lg","bisnis","taro","min","buat","diam",
"gpp","can","tuh","bet","santai","aja","yg", "enggak", "ilu","apaik", "adaail"))

#replace words

docs <- tm_map(docs, gsub, pattern="banayk",replacement="banyak")
docs <- tm_map(docs, gsub, pattern="yg",replacement="yang")
docs <- tm_map(docs, gsub, pattern="bsa",replacement="bisa")
docs <- tm_map(docs, gsub, pattern="bnyk",replacement="banyak")
docs <- tm_map(docs, gsub, pattern="bs",replacement="bisa")
docs <- tm_map(docs, gsub, pattern="mntp",replacement="mantap")
docs <- tm_map(docs, gsub, pattern="jd",replacement="jadi")
docs <- tm_map(docs, gsub, pattern="tidk",replacement="tidak")
docs <- tm_map(docs, gsub, pattern="tdak",replacement="tidak")
docs <- tm_map(docs, gsub, pattern="gmn",replacement="gimana")
docs <- tm_map(docs, gsub, pattern="bg",replacement="bang")
docs <- tm_map(docs, gsub, pattern="jngn",replacement="jangan")
docs <- tm_map(docs, gsub, pattern="kmn",replacement="kemana")
docs <- tm_map(docs, gsub, pattern="kmna",replacement="kemana")
docs <- tm_map(docs, gsub, pattern="lg",replacement="lagi")
docs <- tm_map(docs, gsub, pattern="udh",replacement="sudah")
docs <- tm_map(docs, gsub, pattern="knp",replacement="kenapa")
docs <- tm_map(docs, gsub, pattern="blm",replacement="belom")
docs <- tm_map(docs, gsub, pattern="ig",replacement="instagram")
docs <- tm_map(docs, gsub, pattern="mlah",replacement="malah")
docs <- tm_map(docs, gsub, pattern="ap",replacement="apa")
docs <- tm_map(docs, gsub, pattern="ad",replacement="ada")
docs <- tm_map(docs, gsub, pattern="lgi",replacement="lagi")
docs <- tm_map(docs, gsub, pattern="maen",replacement="main")
docs <- tm_map(docs, gsub, pattern="jg",replacement="juga")
docs <- tm_map(docs, gsub, pattern="knpa",replacement="kenapa")
docs <- tm_map(docs, gsub, pattern="aj",replacement="aja")
docs <- tm_map(docs, gsub, pattern="dri",replacement="dari")
docs <- tm_map(docs, gsub, pattern="dr",replacement="dari")
docs <- tm_map(docs, gsub, pattern="bs",replacement="bisa")
docs <- tm_map(docs, gsub, pattern="dlu",replacement="dulu")
docs <- tm_map(docs, gsub, pattern="dl",replacement="dulu")
docs <- tm_map(docs, gsub, pattern="kalo",replacement="kalau")
docs <- tm_map(docs, gsub, pattern="smngt",replacement="semangat")
docs <- tm_map(docs, gsub, pattern="lbh",replacement="lebih")
docs <- tm_map(docs, gsub, pattern="kyak",replacement="kayak")
docs <- tm_map(docs, gsub, pattern="kyk",replacement="kayak")
docs <- tm_map(docs, gsub, pattern="td",replacement="tadi")
docs <- tm_map(docs, gsub, pattern="skrng",replacement="sekarang")
docs <- tm_map(docs, gsub, pattern="ngk",replacement="enggak")
docs <- tm_map(docs, gsub, pattern="ttp",replacement="tetap")
docs <- tm_map(docs, gsub, pattern="bgt",replacement="banget")
docs <- tm_map(docs, gsub, pattern="bngt",replacement="banget")
docs <- tm_map(docs, gsub, pattern="menangg",replacement="menang")
docs <- tm_map(docs, gsub, pattern="dpn",replacement="depan")
docs <- tm_map(docs, gsub, pattern="klh",replacement="kalah")
docs <- tm_map(docs, gsub, pattern="skrg",replacement="sekarang")
docs <- tm_map(docs, gsub, pattern="brp",replacement="berapa")
docs <- tm_map(docs, gsub, pattern="krna",replacement="karena")
docs <- tm_map(docs, gsub, pattern="pnya",replacement="punya")
docs <- tm_map(docs, gsub, pattern="jls",replacement="jelas")
docs <- tm_map(docs, gsub, pattern="laen",replacement="lain")
docs <- tm_map(docs, gsub, pattern="makasi",replacement="terima kasih")
docs <- tm_map(docs, gsub, pattern="mks",replacement="terima kasih")
docs <- tm_map(docs, gsub, pattern="pake",replacement="pakai")
docs <- tm_map(docs, gsub, pattern="mna",replacement="mana")
docs <- tm_map(docs, gsub, pattern="bgtu",replacement="begitu")
docs <- tm_map(docs, gsub, pattern="dapaat",replacement="dapat")

#Build a term-document matrix

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 15)

#Generate the Word cloud

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 0.5,
          max.words=100, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

#Explore frequent terms and their associations

findFreqTerms(dtm, lowfreq = 4)

#Find related words

v<-as.list(findAssocs(dtm, terms = c("kalah","buruk","acuh","mati","keras","marah"),
                      corlimit = c(0.15,0.15,0.15,0.15,0.15,0.15)))
v

#Find related words (one by one)

v<-as.list(findAssocs(dtm, terms =c("aplikasi"),
                      corlimit = c(0.15)))
View(v$aplikasi)

#barplot

k<-barplot(d[1:15,]$freq, las = 2, names.arg =
             d[1:15,]$word,cex.axis=1.2,cex.names=1.2,
           main ="Most frequent words",
           ylab = "Word frequencies",col = terrain.colors(20))
termFrequency <- rowSums(as.matrix(dtm))
termFrequency <- subset(termFrequency, termFrequency>=115)

text(k,sort(termFrequency, decreasing = T)-
       2,labels=sort(termFrequency, decreasing = T),pch = 6, cex = 1)

