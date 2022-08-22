# Get nice, ordered correlation matrix heatmap 


library(corrplot)
library(tidyverse)

data_raw <- read.csv("~/Research/reynolds_scales_project/data/processed/df_complete.csv")
data_clean <- data[,3:(ncol(data)-1)]
corrmatrix <- data_clean %>% cor(use="complete.obs")

# Bad example
corrplot(corrmatrix, 
         method = "color", 
         addCoef.col="black", 
         #order = "FPC", 
         number.cex= 0.5, 
         tl.cex=0.8, 
         tl.col = "black", 
         type = 'lower', 
         cl.pos = 'b',
         title = "Pearson correlation coefficients to 2dp", 
         mar=c(0,0,1,0)
         )

# Good example
corrplot(corrmatrix, 
         method = "color", 
         addCoef.col="black", 
         order = "FPC", 
         number.cex= 0.5, 
         tl.cex=0.8, 
         tl.col = "black", 
         type = 'lower', 
         cl.pos = 'b',
         title = "Pearson correlation coefficients to 2dp", 
         mar=c(0,0,1,0)
)
