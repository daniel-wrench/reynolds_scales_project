# Get nice, ordered correlation matrix heatmap 

library(corrplot)
library(tidyverse)
library(lares)

# Session -> Set Working Directory -> To Source File Location
# e.g.
setwd("~/Research/Code repos/reynolds_scales_project")

data_raw <- read.csv("data/processed/wind_database.csv")

data <- data_raw |> select(-c(Timestamp, ttu_std, ttc_std, ttk))
data <- data |> relocate(c(Re_lt, Re_di, ttc, ttu, tce, tcf, tci))
corrmatrix <- data |>  cor(use="complete.obs")

# Ordered according to variable order
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

# Ordered according to correlation
# corrplot(corrmatrix, 
#          method = "color", 
#          addCoef.col="black", 
#          order = "FPC", 
#          number.cex= 0.5, 
#          tl.cex=0.8, 
#          tl.col = "black", 
#          type = 'lower', 
#          cl.pos = 'b',
#          title = "Pearson correlation coefficients to 2dp", 
#          mar=c(0,0,1,0)
# )

corr_cross(data,
           #max_pvalue = 0.05, # display only significant correlations (at 5% level)
           top = 20
)

corr_var(data, ttc)
corr_var(data, Re_lt)
corr_var(data, Re_di)
