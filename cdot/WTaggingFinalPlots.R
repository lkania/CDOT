# Load the decorrelated data set for the plots in the paper
load("WTaggingDecorrelated.RData")
# Load necessary libraries
library(ggplot2)
library(ggpubr)
library(latex2exp)

# Setting a theme for the plots
my_theme = theme(plot.title = element_text( size = 15),
                 axis.title = element_text(size = 14),
                 axis.text = element_text(size = 12),
                 legend.title = element_text(size = 14),
                 legend.text = element_text(size = 12),
                 strip.text = element_text(size = 12),
                 legend.position = c(0.85, 0.8))


# Introduction Plots
# Density of invariant mass
plot.mass = ggplot(Test, aes(x = mass, fill = label, colour = label)) + 
  geom_histogram(aes(y = ..density..), 
                 alpha = 0.5, bins = 100, position="identity") +
  scale_color_manual(name = "Class",
                     values = c("0" = "blue",
                                "1" = "darkgrey"),
                     labels = c('Signal', 'Background')) +
  scale_fill_manual(name = "Class",
                    values = c("0" = "blue",
                               "1" = "darkgrey"),
                    labels = c('Signal', 'Background')) +
  ylab("Density")  + 
  xlab("Invariant Mass") +
  ggtitle("Density of Mass") +
  my_theme 

# Density of invariant mass after cuts on classifier output (h)
plot.masscut = ggplot(subset(Test, h > 0.5), 
                      aes(x = mass, fill = label, colour = label)) + 
  geom_histogram(aes(y = ..density..), 
                 alpha = 0.5, bins = 100, position="identity") +
  scale_color_manual(name = "Class",
                     values = c("0" = "blue",
                                "1" = "darkgrey"),
                     labels = c('Signal', 'Background')) +
  scale_fill_manual(name = "Class",
                    values = c("0" = "blue",
                               "1" = "darkgrey"),
                    labels = c('Signal', 'Background')) +
  ylab("Density")  + 
  xlab("Invariant Mass") +
  ggtitle("Density of Mass given classifier h > 0.5") +
  my_theme

# Density of invariant mass after cuts on decorrelated classifier output (T_M(h))
plot.masscutCDOT = ggplot(subset(Test, Trans_h > 0.5), 
                          aes(x = mass, fill = label, colour = label)) + 
  geom_histogram(aes(y = ..density..), 
                 alpha = 0.5, bins = 100, position="identity") +
  scale_color_manual(name = "Class",
                     values = c("0" = "blue",
                                "1" = "darkgrey"),
                     labels = c('Signal', 'Background')) +
  scale_fill_manual(name = "Class",
                    values = c("0" = "blue",
                               "1" = "darkgrey"),
                    labels = c('Signal', 'Background')) +
  ylab("Density")  + 
  xlab("Invariant Mass") +
  ggtitle(TeX("Density of Mass given $T_M(h) > 0.5$")) +
  my_theme 


# Saving the plots
ggsave(plot.mass, 
       filename = paste0("PlotMass.png"),
       width = 6, height = 3.5)

ggsave(plot.masscut, 
       filename = paste0("PlotMassCut.png"),
       width = 6, height = 3.5)

ggsave(plot.masscutCDOT, 
       filename = paste0("PlotMassCutCDOT.png"),
       width = 6, height = 4)



# No Decorrelation plots after cuts on just classifier (h)

plot = list(length = 4)

for(i in 1:4){
  df = subset(Test, h >= (i - 1)*0.25 & h <= i*0.25)
  title = paste0((i-1)*0.25, " \u2264 h \u2264 ", i*0.25)
  
  plot[[i]] = ggplot(df, aes(x = mass, fill = label, colour = label)) + 
    geom_histogram(aes(y = ..density..), 
                   alpha = 0.5, bins = 50, position="identity") +
    lims(x = c(0,1), y = c(0,15)) + 
    ggtitle(title) + 
    scale_color_manual(name = "Class",
                       values = c("0" = "blue",
                                  "1" = "darkgrey"),
                       labels = c('Signal', 'Background')) +
    scale_fill_manual(name = "Class",
                      values = c("0" = "blue",
                                 "1" = "darkgrey"),
                      labels = c('Signal', 'Background')) +
    ylab("Density")  + 
    xlab("Mass") +
    my_theme 
}

ggsave(ggarrange(plot[[1]], plot[[2]], plot[[3]], plot[[4]], 
                nrow = 1, ncol = 4, common.legend = TRUE, legend="bottom"), 
       filename = paste0("WTaggingNoDecorrelation.pdf"),
       width = 12, height = 3.3)

# After decorrelation plots applying cuts on transformed decorrelated classifier

plot1 = list(length = 4)
plot2 = list(length = 4)
title = list(length = 4)
title[[1]] = TeX("0 $\\leq T_M(h) \\leq 0.25$")
title[[2]] = TeX("0.25 $\\leq T_M(h) \\leq 0.5$")
title[[3]] = TeX("0.5 $\\leq T_M(h) \\leq 0.75$")
title[[4]] = TeX("0.75 $\\leq T_M(h) \\leq 1$")

for(i in 1:4){
  df = subset(Test, Trans_h >= (i - 1)*0.25 & Trans_h <= i*0.25)
  #title = paste0((i-1)*0.25, " \u2264 T (h) \u2264 ", i*0.25)
  
  plot1[[i]] = ggplot(df, aes(x = mass, fill = label, colour = label)) + 
    geom_histogram(aes(y = ..density..), 
                   alpha = 0.5, bins = 50, position="identity") +
    lims(x = c(0,1), y = c(0,15)) + 
    ggtitle(title[[i]]) + 
    scale_color_manual(name = "Class",
                       values = c("0" = "blue",
                                  "1" = "darkgrey"),
                       labels = c('Signal', 'Background')) +
    scale_fill_manual(name = "Class",
                      values = c("0" = "blue",
                                 "1" = "darkgrey"),
                      labels = c('Signal', 'Background')) +
    ylab("Density")  + 
    xlab("Mass") +
    my_theme 
  
  plot2[[i]] = ggplot(df, aes(x = mass, fill = label, colour = label)) + 
    geom_histogram(alpha = 0.5, bins = 50, position="identity") +
    #lims(x = c(0,1), y = c(0,13)) + 
    ggtitle(title[[i]]) + 
    scale_color_manual(name = "Class",
                       values = c("0" = "blue",
                                  "1" = "darkgrey"),
                       labels = c('Signal', 'Background')) +
    scale_fill_manual(name = "Class",
                      values = c("0" = "blue",
                                 "1" = "darkgrey"),
                      labels = c('Signal', 'Background')) +
    ylab("Counts")  + 
    my_theme
}

ggsave(ggarrange(plot1[[1]], plot1[[2]], plot1[[3]], plot1[[4]], 
                    plot2[[1]], plot2[[2]], plot2[[3]], plot2[[4]],
                    nrow = 2, ncol = 4, common.legend = TRUE, legend="bottom"), 
         filename = paste0("WTaggingWithDecorrelation.png"),
         width = 12, height = 6.5)






