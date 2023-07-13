###This code does two main things:
#### (a) Loads in the 250 images for a specified image shape and 
#######computes the average WECT and its SD for one direction
#### (b) Looks at the simplex counts and computes the expected WEC
#######computes the average WECT and its SD for one direction

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
shape_name = c("clusters") ##annulus, circle, tetris, square, inverted_small_squares, small_squares
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

##----------------------Plot of SWECT Mean +/- SD

##This computes the average and standard deviation for the WECF. and
###prepares the tibble for plotting
which_dir = "d1" ##Can pick a different direction d1, d2, ..., d8
mean_df = df %>% 
  mutate(weight_dist = case_when(
    weight=="uniform"~"U(0,1)",
    weight=="n17"~"N(0.50,0.17)",
    weight=="n25"~"N(0.50,0.25)",
    weight=="n50"~"N(0.50,0.50)"
  )) %>%
  select(c(1:4, weight_dist, which_dir)) %>%
  pivot_wider(names_from=rep, values_from=which_dir) %>%
  rowwise() %>%
  mutate(d_mean = mean(c_across(5:254)),
         d_sd = sd(c_across(5:254))) %>% 
  ungroup()

### Square expectation avg
sqavgexp <- data.frame(x = seq(-45, 45, by = 1))
sqavgexp$y <- ifelse(sqavgexp$x > -17, 0.5, 0)

###Plot of average WECF +/- SD
alpha0=.1 ##Transparency of bands around averages
sd_factor = 1 #sqrt(num_rep)
mean_df %>%
  ggplot(aes(x=t, y=d_mean)) +
  geom_ribbon(mean_df%>%filter(weight=="uniform"), mapping=aes(ymax=d_mean+d_sd/sd_factor, ymin=d_mean-d_sd/sd_factor), alpha=alpha0, fill="purple") +
  geom_ribbon(mean_df%>%filter(weight=="n17"), mapping=aes(ymax=d_mean+d_sd/sd_factor, ymin=d_mean-d_sd/sd_factor), alpha=alpha0, fill="red") +
  geom_ribbon(mean_df%>%filter(weight=="n25"), mapping=aes(ymax=d_mean+d_sd/sd_factor, ymin=d_mean-d_sd/sd_factor), alpha=alpha0, fill="green") +
  geom_ribbon(mean_df%>%filter(weight=="n50"), mapping=aes(ymax=d_mean+d_sd/sd_factor, ymin=d_mean-d_sd/sd_factor), alpha=alpha0, fill="blue") +
  geom_line(aes(color=weight_dist), linewidth=1) +
#   geom_line(data=sqavgexp, aes(x=x, y=y), linetype="dashed", color="black") + 
  labs(x=expression(t),
       y="WECT",
       color="Distribution",
       linetype="Distribution") +
  theme_minimal() +
  theme(text = element_text(size=40),
        legend.title = element_blank(),
        legend.position=c(.8,.15),
        legend.background = element_blank(),
        legend.box.background = element_rect(color = "black"),
        legend.key.size = unit(.95, 'cm')) #+
    # coord_cartesian(xlim = c(-14,15))

ggsave(
  "wect-images/clusters_avgfe_dir0.pdf",
  plot = last_plot(),
  device = "pdf",
)

###Get average WEC using the largest filtration value (t=1) with the uniform weights
mean_df %>%
  filter(weight == "uniform") %>%
  select(t, d_mean, d_sd)%>%
  slice(n_grid)

##Empirical average WEC (at t=1)
empirical_avg_wec = mean_df %>%
  filter(weight == "uniform") %>%
  select(t, d_mean, d_sd)%>%
  .[n_grid, 2]
(empirical_avg_wec)

##Empirical sd WEC (at t=1)
empirical_sd_wec = mean_df %>%
  filter(weight == "uniform") %>%
  select(t, d_mean, d_sd)%>%
  .[n_grid, 3]
(empirical_sd_wec)

##Empirical standard error WEC (at t=1)
###This is what get used in a confidence interval for the expected value
empirical_se_wec = empirical_sd_wec/sqrt(num_rep)
empirical_se_wec



########################################################################
###########################Count simplexes
########################################################################

##UPDATE:  specify your working directory
wd_data_simp = "/Users/jessi/Dropbox/aaa_tda/galaxy-shape/shape_wect_lama/"

##Load data
simp0 = read_csv(str_c(wd_data_simp, "count_simplices_annulus.csv"), 
                 col_names = c("t", "d0sum", "d1sum", "d2sum"))

##Build tibble that includes the expected WEC using the average (avg0) and maximum (max0) extensions
###I also added the ECT (ect)
###The expected WECs assume the pixel intensity distribution is centered at 0.5 (for avg0),
###and is U(0,1) (for max0)
simp = simp0 %>%
  mutate(avg0 = 0.5*(d0sum-d1sum+d2sum),
         max0 = (1/2)*d0sum-(2/3)*d1sum+(3/4)*d2sum,
         ect = d0sum-d1sum+d2sum)

##Quick look at the original data (# of simplexes by dimension vs t)
simp %>%
  ggplot(aes(x=t)) +
  geom_point(aes(y=d0sum, color="Points"), size=2) +
  geom_point(aes(y=d1sum, color="Segments"), size=2) +
  geom_point(aes(y=d2sum, color="Triangles"), size=2) +
  xlab(expression(t)) +
  ylab("# of Simplexes") + 
  theme_minimal() +
  theme(text = element_text(size=40),
        legend.title = element_blank(),
        legend.position=c(.25,.75),
        legend.background = element_blank(),
        legend.box.background = element_rect(color = "black"),
        legend.key.size = unit(1.25, 'cm')) 


###Look at largest filtration parameter value to compute expected WEC
(nn = nrow(simp)) ##Number of rows in data matrix
simp %>%
  slice(nn)
(n0 = simp$d0sum[nn]) ##Number of 0-simplexes in final row (largest t)
(n1 = simp$d1sum[nn]) ##Number of 1-simplexes in final row (largest t)
(n2 = simp$d2sum[nn]) ##Number of 2-simplexes in final row (largest t)

## average extension
0.5*(n0 - n1 + n2)
simp %>% slice(nn) %>% select(t, avg0)
### 0.5

## max extension
(max_ext = (1/2)*n0 - (2/3)*n1 + (3/4)*n2)
simp %>% slice(nn) %>% select(t, max0)
### -20.5

###The above has t=4, but around t=1 is the same
simp %>%
  filter(between(t, .98, 1.03))


##CI for the expected WEC using the 250 images at the max filtration parameter 
conf_level = 0.95 ##Confidence level for interval
tibble(lower_ci = as.numeric(empirical_avg_wec - qnorm((1-conf_level)/2)*empirical_se_wec),
       upper_ci = as.numeric(empirical_avg_wec + qnorm((1-conf_level)/2)*empirical_se_wec),
       empirical_avg = as.numeric(empirical_avg_wec),
       expected_wec = max_ext)

###Note:  the expected WEC is not within the CI

