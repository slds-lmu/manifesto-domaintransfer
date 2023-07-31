###################################################### 
###### Data Extraction of Manifesto Project ##########
###################################################### 
# As described in the following you need to create your own API key
# on the website to get access to the Manifesto Project Database


### packages
# install package manifestoR and connect to API to get access
# install.packages("manifestoR")
library(manifestoR)
mp_setapikey("./Coding/manifesto_apikey.txt")
# Note to reproduce code: Create your own key on website (https://manifesto-project.wzb.eu/) 
# by signing in on website and then get your own API key on your profile

## other libraries for data cleaning and plots
library(tidyverse)
library(ggplot2)


### set version !!!!!!!!!!!!!!
# 2018-2 or 2021-1
# version <- "2018-2"
version <- "2021-1"

# set version to dowload right corpus
# and  codebook for categories 
if (version == "2018-2") {
  # use version 2018-2 from paper  
  mp_use_corpus_version("2018-2")
  categories = mp_codebook(version = "MPDS2018b")
  print(paste("set version to", version))
  } else {
    # or use latest version: 2021-1
    mp_use_corpus_version("2021-1")
    categories = mp_codebook(version = "current")
    print(paste("set version to", version))
}


# check
mp_which_corpus_version()




####################################################
### download text and corresponding categories with additional information 
### extracted variables: countryname, text, language, code, document_index, date, topic_8 
####################################################
# note: codes unequal to 8 categories (more codes than 8)
# codes need to be looked up in mp_codebook to allocate corresponding category


## initialization variables
# list of selected countries (see paper)
countries <- c("Australia", "Canada", "Ireland", "New Zealand",  
               "United Kingdom", "United States", "South Africa")

# data frame: one column with text 
# (= quasi-sentences for every document for every selected country)
corpra_country = NULL

# language of each text
language = NULL

# country of origin of each text
countryname = NULL

# codes of each text
copra_codes = NULL

# enumerates document of each country
document_index = NULL

# publication date of document
date <- NULL


## looping over 7 countries [i] and every document [i] per country
for (j in seq_along(countries)){
# extract text documents per country
documents_country <- mp_corpus(countryname == countries[j])

  for (i in seq_along(documents_country)) {
    # make one data frame for text
    text_per_document = data.frame(text = content(documents_country[[i]]))
    corpra_country = bind_rows(corpra_country, text_per_document)
  
    # creat corresponding language vector (keeps track of language of each document)
    # rep(): to keep track per quasi-sentence
    language = append(language, rep(meta(documents_country[[i]])$language, length(content(documents_country[[i]]))))
  
    # creat corresponding countryname vectorn (keeps track of countryname of each document)
    # rep(): to keep track per quasi-sentence
    countryname = append(countryname, rep(countries[j], length(content(documents_country[[i]]))))
    
    # vector of which document
    document_index = append(document_index, rep(i, length(content(documents_country[[i]]))))
    
    # extract date of documtet
    date = append(date, rep(documents_country[[i]]$meta$date, length(content(documents_country[[i]]))))
  } # end i loop

# codes extracted on document level 
# collect codes for every quasi-sentence in copra_codes vector
copra_codes = append(copra_codes, codes(documents_country))
}# end j loop


## check dimension: all the same
dim(corpra_country) # text
length(language)
length(countryname)
length(copra_codes)
length(document_index)
length(date)


## merge variable to one data frame copra_per_country 
copra_per_country = bind_cols("countryname" = countryname, corpra_country, 
                              "language" =language, "code" = copra_codes, 
                              "document_index" = document_index, "date" = date)

## check dim and colnames
dim(copra_per_country)
colnames(copra_per_country)

## check descriptive/ categories of each variable
# language
table(copra_per_country$language, useNA = c("always"))

# countryname: observations per country
table(copra_per_country$countryname, useNA = c("always"))
#copra_per_country %>% group_by(countryname)  %>% count() 

# document index: quasi sentencses per document
#table(copra_per_country$document_index, useNA = c("always")) # wrong as index is per country
overview_doc_i <- copra_per_country %>% group_by(countryname) %>% count(document_index)
colnames(overview_doc_i)[3] <- "quasi_sentences"
summary(overview_doc_i$quasi_sentences)


# date
table(copra_per_country$date, useNA = c("always"))
summary(copra_per_country$date)
hist(copra_per_country$date)


# code
table(copra_per_country$code, useNA = c("always"))# note NA and 000; 000 = "no topic"







##### add categories
## categories are saved in mp_codebook()
# depends on version of data (see version in beginning)
dim(categories)
sapply(categories, n_distinct)
# Note: 8 domain_name equals variable topic of paper

# compare codes in text and in category table
sort(categories$code)
sort(unique(copra_per_country$code))

# 3 uncategorizeed codes in data
# categories$code[!(categories$code %in% unique(copra_per_country$code))] not important
unique(copra_per_country$code)[!(unique(copra_per_country$code) %in% categories$code)] # relevant
nrow(copra_per_country %>% filter(code == "H" ))
nrow(copra_per_country %>% filter(code == "0" ))
nrow(copra_per_country %>% filter(code == "H" |code == "0" ))
nrow(copra_per_country %>% filter(is.na(code)))
# From Manifesto project:
#'H' marks sentences that have served as some form of heading in the manifesto, 
# those sentences thus help to document the underlying structure of the election program. 
# labels they can simply be understood as 'NA


## Handle 3 uncategorized codes 
# make H categories to NA
# -> This will automatically happen in join below

# make 0 categories to "000"
copra_per_country[!is.na(copra_per_country$code) & copra_per_country$code == "0", "code"] <- "000"
nrow(copra_per_country %>% filter(code == "0" ))

# check NA before join
#nrow(copra_per_country %>% filter(is.na(code)))
#sum(rowSums(is.na(copra_per_country)))
colSums(is.na(copra_per_country))

# join/ mergen codes (categories) and text data frame (copra_per_country)
join = left_join(copra_per_country, categories[, c("code", "domain_name")], by = "code")


## check join
# dimension and columns
dim(join)
dim(copra_per_country)
colnames(join)

# example
join[500,]
copra_per_country[500,]

# look at NA and 000; 000 = "no topic"
nrow(join %>% filter(code == "000"))
nrow(join %>% filter(is.na(code)))

nrow(join %>% filter((domain_name == "NA")))
nrow(join %>% filter(is.na(domain_name)))
# check NA
nrow(join %>% filter(code == "H" )) + nrow(join %>% filter(is.na(code))) == nrow(join %>% filter(is.na(domain_name)))


## filter english
not_english <- join %>% filter(language != "english")
nrow(not_english)
final_corpus = join %>% filter(language == "english")
dim(final_corpus)




## Investigating NAs in categories (variable domain_name)
nrow(final_corpus %>% filter(domain_name == "NA"))
sum(is.na(final_corpus$domain_name))
unique(final_corpus$domain_name)


# Note about NAs:
# There are two kind of NAs
#   1. "NA"/ not topic/ code 000 category as class in following calssification analysis (not delete)
#   2. NA per quasi-sentence (delete)
# next step: domain_name without NA, but with no topic:


# without NA
final_corpus_na <- final_corpus[!is.na(final_corpus$domain_name),]
sum(is.na(final_corpus_na$domain_name))
dim(final_corpus_na)
unique(final_corpus_na$domain_name)

# "NA" as "no topic"
final_corpus_na[final_corpus_na$domain_name == "NA", "domain_name"] <- "no topic"
dim(final_corpus_na)
unique(final_corpus_na$domain_name)
nrow(final_corpus_na %>% filter(domain_name == "no topic"))

# lowercase version of domain_name to compare with topic_8 of paper
final_corpus_na$domain_name <- tolower(final_corpus_na$domain_name)

# also rename variable domain_name to topic_8
final_corpus_na <- rename(final_corpus_na,  topic_8 = domain_name)
colnames(final_corpus_na)
dim(final_corpus_na)



### save data as csv
#if (version == "2018-2") {
### for version 2018
#write.csv(final_corpus_na, "./data/source_corpus_2018.csv", row.names=FALSE,fileEncoding="UTF-8")
#write.csv(final_corpus_na, "C:/Users/nadja/OneDrive/Topic_Classification/JupyterLab/data/source_corpus_2018.csv", row.names=FALSE,fileEncoding="UTF-8")
#} else {### for version 2021
#write.csv(final_corpus_na, "./data/source_corpus_2021.csv", row.names=FALSE,fileEncoding="UTF-8")
#write.csv(final_corpus_na, "C:/Users/nadja/OneDrive/Topic_Classification/JupyterLab/data/source_corpus_2021.csv", row.names=FALSE, fileEncoding="UTF-8")
#}
