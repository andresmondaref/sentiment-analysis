# Set working directory
setwd("D:/SKRIPSI ANDRES")

#	KESELURUHAN KONTEN 	
# read in the data install.packages ("readr") library(readr)
analisis1 <- read.csv("data_cleaning2.csv",header = TRUE)

## (LANGKAH 1) Script R Preprocessing Data dengan Text Mining
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
# Convert the text to lower case
docs <- tm_map(docs, content_transformer (tolower))
#Remove punctuation
docs <- tm_map(docs, toSpace, "[[:punct:]]")
#Remove numbers
docs <- tm_map(docs, toSpace, "[[:digit:]]")
# add two extra stop words: "available" and "via"
myStopwords = readLines("stopwordsid.csv")
# remove stopwords from corpus
docs <- tm_map(docs, removeWords, myStopwords)
# Remove your own stop word

# specify your stopwords as a character vector
docs <- tm_map(docs, removeWords, 
               c("dan", "yang", "untuk", "dari","lagi", "kami","jika", "pun", "selain", 
                 "tersebut", "serta","karena", "bersama", "pada", "itu", "dalam", "di",
                 "juga", "melalui", "menjadi", "melakukan", "adalah", "jadi", "saja", 
                 "ingin", "selengkapnya", "tentang", "hal", "bisa", "sahabatdikbud", "tetap", 
                 "tapi", "hanya", "yaitu", "bagi", "seperti", "suatu", "ini", "dapat", 
                 "sebagai", "harus", "sesuai","dengan", "akan", "per", "ada", "telah", 
                 "para", "masa", "tidak", "secara", "atau", "tak", "oleh", "kita","apa", 
                 "agar", "sudah","kepada", "saat", "yukks","wkwk", "bgt","min",
                 "tpi", "weh", "loh", "tik","klo", "juga","nah", "aja","jdi",
                 "iya", "dpt", "jok", "nya", "kah","for","pst","ntr"))
#REMOVE EMOJI


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

#wordcloud
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



#(Langkah 2) Script R Pelabelan dan Pembobotan
docs<-read.csv("data_cleaning2.csv",header=TRUE)

#skoring

positif <- scan("D:/SKRIPSI ANDRES/positif.txt",what="character",
                comment.char=";")
negatif <- scan("D:/SKRIPSI ANDRES/negatif.txt",what="character",
                comment.char=";")
score.sentiment = function(docs, kata.positif, kata.negatif, .progress='none')
  
{
  require(plyr)
  require(stringr)
  scores = laply(docs,function (kalimat, kata.positif, kata.negatif) {
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
    score = sum(positif.matches) - (1*sum(negatif.matches))
    return(score)
    },
    kata.positif, kata.negatif, .progress=.progress )
    scores.df = data.frame( score=scores, text=docs )
    return(scores.df)
}

hasil = score.sentiment(docs$text, kata.positif, kata.negatif)
View(hasil)
#
hasil$klasifikasi<- ifelse(hasil$score<0, "Negatif","Positif")
hasil$klasifikasi
View(hasil)





