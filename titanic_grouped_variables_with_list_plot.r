# import library
library(dplyr)
library(Hmisc)
library(mice)
library(corrplot)
library(VIM)
library(caret)
library(ggplot2)
library(gridExtra)
# ------------------------------------------------------- #

# import data
train <- read.csv("E:/Data_and_Script/Titanic Data/train.csv",stringsAsFactors = FALSE)
test <- read.csv("E:/Data_and_Script/Titanic Data/test.csv",stringsAsFactors = FALSE)

# merge data and glimpse
test$Survived <- NA
df <- rbind(train,test)
df[df == ""] <- NA

str(df)
summary(df)
describe(df)

# ------------------------------------------------------- #
# base line model with gbm
set.seed(13)
bl_idx <- createDataPartition(train$PassengerId,p = .75,list = FALSE,times = 1)
bl_train <- train[bl_idx,]
bl_test <- train[-bl_idx,]
model_bl <- train(factor(Survived) ~ factor(Pclass)+factor(Sex)+Age+SibSp+Parch+Fare+factor(Embarked),data = bl_train,
                  method = 'rf',na.action = na.omit)
max(model_bl$results$Accuracy) # 0.816569

# ------------------------------------------------------- #
# data pre-processing


# fix some bugs
df$SibSp[df$PassengerId==280] = 0
df$Parch[df$PassengerId==280] = 2
df$SibSp[df$PassengerId==1284] = 1
df$Parch[df$PassengerId==1284] = 1

# NA check numeric
md.pattern(df)
# VIM PACKAGE
aggr_plot <- aggr(df, col=c('navyblue','red'), prop = TRUE,cex.numbers = 0.7,
                  numbers=TRUE, sortVars=TRUE, labels=names(data), 
                  cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))


# correlation
df %>% select_if(is.numeric) %>% cor(use="pairwise.complete.obs") %>% 
    corrplot(method = "number",type="upper",tl.srt=30,tl.col="black")

# ------------------------------------------------------- #
# imputation

# Fare
describe(df$Fare)
# impute fare with sibsp,pclass,Parch,sex and embarked in RF
model_fare <- train(Fare ~ Pclass + Sex + Embarked + SibSp + Parch, 
                    data = df[!is.na(df$Fare),],method = 'rf',na.action = na.omit)

df$Fare[is.na(df$Fare)] <- predict(model_fare,df[is.na(df$Fare),] %>% 
                                       select(Pclass , Sex , Embarked , SibSp , Parch))
model_fare$finalModel      # 52% Var explained


# embarked
describe(df$Embarked) # no missing but 2 value none
df[is.na(df$Embarked),]
# imputate with glm
model_embarked <- train(Embarked ~ Pclass + SibSp + Parch + Fare,data = df[!is.na(df$Embarked),], method = "rf",na.action = na.omit)
df$Embarked[is.na(df$Embarked)] <- "C"
predict(model_embarked,df[is.na(df$Embarked),] %>% select(Pclass,SibSp,Parch,Fare))

describe(df$Embarked)

# Cabin
describe(df$Cabin)
# impute NA with X
df$Cabin[is.na(df$Cabin)] <- "X"
describe(df$Cabin)


# cabin captial
df$cpt_canin <- df$Cabin %>% substr(1,1)

# Age
describe(df$Age)
# plot age vs pclass and sibsp
ggplot(df,aes(Pclass,fill = !is.na(Age))) + geom_bar(position = "dodge") +
    labs(title = "Passengers has Age",fill = "Has Age")

# plot survived vs age and pclass 1&2
ggplot(train %>% filter(Pclass !=3),aes(Age,fill = factor(Survived))) +geom_density(alpha = .7)

# setup Minor as age , 14 and Pclass = 1 & 2
df$Minor <-ifelse(df$Age < 14 & df$Pclass != 3,1,0)
df$Minor <- ifelse(is.na(df$Minor),0,df$Minor)

# impute age
model_age <- train(Age ~ Pclass + Sex + Embarked + SibSp + Parch + Fare, 
                    data = df[!is.na(df$Fare),],method = 'rf',na.action = na.omit)
df$new_age <- df$Age
df$new_age[is.na(df$new_age)] <- predict(model_age,df[is.na(df$Age),] %>% select(Pclass,Sex,Embarked,SibSp,Parch,Fare))

# how good is prediction
model_age$finalModel$rsq %>% mean() # 0.3053673

p1 <- ggplot(train,aes(Age,fill = factor(Survived))) + geom_density(alpha = .7) +ggtitle("Age before imputation")
p2 <- ggplot(df[1:891,],aes(new_age,fill = factor(Survived))) +geom_density(alpha = .7) +
    ggtitle("Age after imputation")
grid.arrange(p1, p2, nrow = 1)
# the shape changed. seems this predictio is not good enough


# --------------------------------------------------------------- #
# grouped
# number of groups with same ticket
df$TFreq <- ave(seq(nrow(df)), df$Ticket,  FUN=length)
ggplot(df[1:891,],aes(TFreq,fill = factor(Survived))) + geom_density(alpha = .7) + facet_grid(Sex ~. )
# how many ppl in each cabin
df$ppl_cabin <- ave(seq(nrow(df)),df$Cabin,FUN = length)
ggplot(df[1:891,],aes(ppl_cabin,fill = factor(Survived))) + geom_density(alpha = .7) + facet_grid(Sex ~. )

# grouped by fare
df$FFreq <- ave(seq(nrow(df)),df$Fare,FUN = length)
ggplot(df[1:891,],aes(FFreq,fill = factor(Survived))) + geom_density(alpha = .7) + facet_grid(Sex ~. )

# Title
df$Title <- df$Name %>% sapply(function(x) strsplit(x,(df$Name),split = "[.,]")[[1]][2])

# family name
df$Surname <- df$Name %>% sapply(function(x) strsplit(x,(df$Name),split = "[.,]")[[1]][1])

# find groups by GID
# ------------------------------- #
# We now assign group identifications (GID) to each passenger. The assignment follows the following rules:
    
# The maximum group size is 11.
# First we look for families by Surname and break potentially identical family names by appending a family size.
# Single families by the above rule are labeled ¡®Single¡¯.
# Look at the ¡®Single¡¯ group and assign a GID to those that share a Ticket value.
# Look at the ¡®Single¡¯ group and assign a GID to those that share a Fare value.
# ------------------------------- #
# family name + family size
df$GID  <- paste0(df$Surname,as.character(df$SibSp+df$Parch+1))
df$GID[df$SibSp+df$Parch == 0] <- "Single"

# single ppl with same Ticket
df$GID[df$GID == "Single" & df$TFreq > 1 & df$TFreq < 12] <- df$Ticket[df$GID == "Single" & df$TFreq > 1 & df$TFreq < 12] 

# single ppl with same fare
df$GID[df$GID == "Single" & df$FFreq > 1 & df$FFreq < 12] <- df$Ticket[df$GID == "Single" & df$FFreq > 1 & df$FFreq < 12] 
  

# Pclass and Sex
# ---------------------------------------------------- #
# How to plot in grid
p <- list()
item <- 1;
ylim <- 300;
for (class in c(1:3)) {
    for (sex in c("male","female")) {
        p[[item]] <- ggplot(train %>% filter(Sex == sex,Pclass == class),aes(factor(Survived))) +
                    geom_bar(alpha = .7) + ggtitle(paste0("Pclass = ", Pclass," and Sex = ",sex))
        item <- item + 1
    }
}
do.call(grid.arrange,p)
# ---------------------------------------------------- #

# new model
model_n <- train(factor(Survived) ~ factor(Pclass) + factor(Sex) + new_age + SibSp + Parch + Fare +
                     factor(Embarked) + Minor + TFreq + FFreq + GID, data = df[1:891,],method = "rf")
varImp(model_n)
max(model_n$results$Accuracy)
# 0.8277788
