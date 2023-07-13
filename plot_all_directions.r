###Packages
library(tidyverse)
library(viridis)
library(scales)
library(reticulate)
np <- import("numpy")

########################################################################
###########################Average WECT
########################################################################
## Load data for average WECTs

##UPDATE:  specify your working directory
wd_data = "data/wects_disc/AVGfe/15dir/"

##Parameter:  update as desired
num_directions = 15 #number of directions on the sphere
num_rep = 250 #number of images
shape_name = c("square") ##annulus, circle, tetris, square, inverted_small_squares, small_squares
weight_name = c("uniform", "n17", "n25", "n50")
n_grid = 91 #number of points on the filtration grid
t_grid = seq(-45,45,length.out=n_grid) #the filtration grid

##Set up the data frame
tbl_colnames = c("im_shape", "weight", "t", "rep", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8")
df = as_tibble(matrix(nrow = 0, ncol = length(tbl_colnames)), .name_repair = ~ tbl_colnames) %>%
  mutate_at(3:12, as.numeric) %>%
  mutate_at(1:2, as.character)

##Load the data and add to tibble.  This takes ~30 seconds to run.  Obviously it can be made to run faster!
for(ii in 1:length(shape_name)){ ##Can load more shape names.  Currently only "annulus" is specified
  for(jj in 1:length(weight_name)){ ##Cycle through the four intensity weight distributions
    print(jj)
    ##Load data file
    npz1 <- np$load(str_c(wd_data, shape_name[ii], "/", weight_name[jj], ".npz"))
    for(kk in 1:num_rep){ ##Grab info from each of the 250 images
      df0 = tibble(im_shape = rep(shape_name[ii], n_grid),
                   weight = rep(weight_name[jj], n_grid),
                   t = t_grid,
                   rep = rep(kk, n_grid),
                   d1 = npz1$f[[as.character(kk-1)]][1:91],
                   d2 = npz1$f[[as.character(kk-1)]][91:181],
                   d3 = npz1$f[[as.character(kk-1)]][182:272],
                   d4 = npz1$f[[as.character(kk-1)]][273:363],
                   d5 = npz1$f[[as.character(kk-1)]][364:454],
                   d6 = npz1$f[[as.character(kk-1)]][455:545],
                   d7 = npz1$f[[as.character(kk-1)]][546:636],
                   d8 = npz1$f[[as.character(kk-1)]][637:727])
      df = df %>%
        bind_rows(df0)
    }
  }
}

df %>%
  ggplot() + 
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d1[1:91])) +
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d2[1:91])) +
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d3[1:91])) +
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d4[1:91])) +
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d5[1:91])) +
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d6[1:91])) +
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d7[1:91])) +
  geom_line(data=df%>%filter(im_shape=="square", weight=="uniform", rep==1), mapping=aes(x=t[1:91], y=d1[1:91])) +
  theme_minimal()
