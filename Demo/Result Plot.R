library(ggplot2)

df = data.frame(Category = c('N','N','N','N','N','N','N','Y','N','N'),
                Candidate = c('1','2','3','4','5','6','7','8','9','10'),
                Order = c(1,2,3,4,5,6,7,8,9,10),
                Probability = c(0.12286159,0.01212837,0.02890266,0.24690156,0.26432127,
                                0.00347028,0.0037853,0.31635445,0.00061923,0.0006553 ))

df$Candidate <- factor(df$Candidate, levels = df$Candidate[order(df$Order)])

head(df)



plot <- ggplot(data=df, aes(x=Candidate,y=Probability,fill=Category)) + ggtitle("CSEye Sample Results") + guides(fill=FALSE) + geom_bar(stat="identity")  + theme_minimal() 

plot

