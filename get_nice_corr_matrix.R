# Get nice, ordered correlation matrix heatmap 

library(corrplot)
library(tidyverse)
library(lares)

data_raw <- read.csv("~/Research/reynolds_scales_project/data/processed/df_complete.csv")
data_clean <- data_raw[1:28,2:ncol(data_raw)]
data_clean <- relocate(data_clean, c(taylor_scale, corr_scale_int, corr_scale_exp_fit, corr_scale_exp_trick))
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

corr_cross(data_clean,
           #max_pvalue = 0.05, # display only significant correlations (at 5% level)
           top = 10
)

corr_var(data_clean, taylor_scale)
