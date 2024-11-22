source("./cdot/Requirements.R")
# Loading decorrelation 3b-4b data and required packages
load("./cdot/DecorrelatedData.RData")
library(ggplot2)
library(ggpubr)
library(latex2exp)

# RData file data frames:

# Background.Classifier: Training data for the classifier
# Signal.Classifier: Training signal data for the classifier
# Background.CDOT: Training data for the decorrelation - VALIDATION DATA
# Background.Test.3b: Test 3b data
# Background.Test.4b: Test 4b data
# Signal.Test: Test signal data


# Plots of the Decorrelation Algorithm:

my_theme = theme(plot.title = element_text(size = 15),
                 axis.title = element_text(size = 14),
                 axis.text = element_text(size = 12),
                 legend.title = element_text(size = 14),
                 legend.text = element_text(size = 12),
                 strip.text = element_text(size = 12),
                 legend.position = c(.95, .95),
                 legend.justification = c("right", "top"))

# EDA plots - Histogram of all variables

library(reshape2)
Background.3b = rbind(Background.Test.3b[, 1:16],
                      Background.CDOT[, 1:16],
                      Background.Classifier[, 1:16])
Background.4b = Background.Test.4b[, 1:16]
Signal = rbind(Signal.Classifier[, 1:16], Signal.Test[, 1:16])

mpgid <- mutate(Background.3b, id = as.numeric(rownames(Background.3b)))
mpgstack <- melt(mpgid, id = "id")
pp.3b <- ggplot(aes(x = value), data = mpgstack) +
  geom_histogram(alpha = 0.5, bins = 50) +
  facet_wrap(~variable, scales = "free")
# pp + stat_bin(geom="text", aes(label=..count.., vjust=-1))
ggsave("./cdot/img/3b-histograms.pdf", pp, scale = 1)

mpgid <- mutate(Background.4b, id = as.numeric(rownames(Background.4b)))
mpgstack <- melt(mpgid, id = "id")
pp.3b <- ggplot(aes(x = value), data = mpgstack) +
  geom_histogram(alpha = 0.5, bins = 50) +
  facet_wrap(~variable, scales = "free")
# pp + stat_bin(geom="text", aes(label=..count.., vjust=-1))
ggsave("./cdot/img/4b-histograms.pdf", pp, scale = 1)

mpgid <- mutate(Signal, id = as.numeric(rownames(Signal)))
mpgstack <- melt(mpgid, id = "id")
pp.3b <- ggplot(aes(x = value), data = mpgstack) +
  geom_histogram(alpha = 0.5, bins = 50) +
  facet_wrap(~variable, scales = "free")
# pp + stat_bin(geom="text", aes(label=..count.., vjust=-1))
ggsave("./cdot/img/Signal-histograms.pdf", pp, scale = 1)


sapply(Background.3b, range)
sapply(Background.4b, range)
sapply(Signal, range)


# Decorrelated Plots

# Validation Data - Background.CDOT


plot3b = list(length = 4)
plot4b = list(length = 4)
Test.df = rbind(Background.Test.4b, Signal.Test)
title = list(length = 4)
title[[1]] = TeX("0 $\\leq T_M(h) \\leq 0.25$")
title[[2]] = TeX("0.25 $\\leq T_M(h) \\leq 0.5$")
title[[3]] = TeX("0.5 $\\leq T_M(h) \\leq 0.75$")
title[[4]] = TeX("0.75 $\\leq T_M(h) \\leq 1$")

for (i in 1:4) {
  df_4b = subset(Test.df, Trans_h >= (i - 1) * 0.25 & Trans_h <= i * 0.25)
  #title = paste0((i-1)*0.25, " \u2264 T (h) \u2264 ", i*0.25)


  df_3b = subset(Background.Test.3b, Trans_h >= (i - 1) * 0.25 & Trans_h <= i * 0.25)
  #title = paste0((i-1)*0.25, " \u2264 T (h) \u2264 ", i*0.25)
  #df = subset(Background.CDOT, Trans_h >= cut_offs_CDOT[i])
  #title = paste0("BR:", RR[i], "%, M|T(h) in [", 
  #               round(cut_offs_CDOT[i],2), ",", 1, "]")

  plot4b[[i]] = ggplot(df_4b, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(aes(y = ..density..),
                   alpha = 0.5, bins = 50, position = "identity") +
    lims(x = c(200, 1500), y = c(0, 0.02)) +
    ggtitle(title[[i]]) +
    scale_color_manual(values = c("Signal" = "blueviolet",
                                  "4b" = "darkgrey",
                                  "3b" = "skyblue"),
                       limits = c("3b", "4b", "Signal")) +
    scale_fill_manual(values = c("Signal" = "blueviolet",
                                 "4b" = "darkgrey",
                                 "3b" = "skyblue"),
                      limits = c("3b", "4b", "Signal")) +
    ylab("4b Background Density") +
    xlab("Mass") +
    my_theme

  plot3b[[i]] = ggplot(df_3b, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(aes(y = ..density..),
                   alpha = 0.9, bins = 50, position = "identity") +
    scale_color_manual(values = c("Signal" = "blueviolet",
                                  "4b" = "darkgrey",
                                  "3b" = "skyblue"),
                       limits = c("3b", "4b", "Signal")) +
    scale_fill_manual(values = c("Signal" = "blueviolet",
                                 "4b" = "darkgrey",
                                 "3b" = "skyblue"),
                      limits = c("3b", "4b", "Signal")) +
    lims(x = c(200, 1500), y = c(0, 0.007)) +
    ggtitle(title[[i]]) +
    ylab("3b Background Density") +
    xlab("Mass") +
    my_theme


}

ggsave(ggarrange(plot4b[[1]], plot4b[[2]], plot4b[[3]], plot4b[[4]],
                 plot3b[[1]], plot3b[[2]], plot3b[[3]], plot3b[[4]],
                 nrow = 2, ncol = 4, common.legend = TRUE, legend = "bottom"),
       filename = paste0("./cdot/img/3b4bWithDecorrelation.png"),
       width = 12, height = 6)


# Correlated Validation Plot

for (i in 1:4) {
  df_4b = subset(Test.df, h >= (i - 1) * 0.25 & h <= i * 0.25)
  title = paste0((i - 1) * 0.25, " \u2264 h \u2264 ", i * 0.25)

  plot4b[[i]] = ggplot(df_4b, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(aes(y = ..density..),
                   alpha = 0.5, bins = 50, position = "identity") +
    lims(x = c(200, 1500), y = c(0, 0.023)) +
    ggtitle(title) +
    scale_color_manual(values = c("Signal" = "blueviolet",
                                  "4b" = "darkgrey")) +
    scale_fill_manual(values = c("Signal" = "blueviolet",
                                 "4b" = "darkgrey")) +
    ylab("4b Background Density") +
    xlab("Mass") +
    my_theme
}

ggsave(ggarrange(plot4b[[1]], plot4b[[2]], plot4b[[3]], plot4b[[4]],
                 nrow = 1, ncol = 4, common.legend = TRUE, legend = "bottom"),
       filename = paste0("./cdot/img/4bNoDecorrelation.png"),
       width = 12, height = 3)


# Test 3b data

plot_test = list(length = length(RR))
Test = rbind(Background.Test.3b, Signal.Test)
Test$Class = factor(Test$Class, levels = c("Signal", "3b"))

for (i in 1:length(RR)) {
  df = subset(Test, Trans_h >= cut_offs_CDOT[i])
  title = paste0("BR:", RR[i], "%, M|T(h) in [",
                 round(cut_offs_CDOT[i], 2), ",", 1, "]")

  plot_test[[i]] = ggplot(df, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(aes(y = ..density..),
                   alpha = 0.5, bins = 50, position = "identity") +
    lims(x = c(200, 1500)) +
    ggtitle(title) +
    scale_color_manual(values = c("3b" = "seagreen3",
                                  "Signal" = "blue")) +
    scale_fill_manual(values = c("3b" = "seagreen3",
                                 "Signal" = "blue")) +
    ylab("Density") +
    xlab("Mass") +
    my_theme +
    theme(legend.position = "none")
}

ggsave(grid.arrange(plot_test[[1]], plot_test[[2]], plot_test[[3]],
                    plot_test[[4]], plot_test[[5]], plot_test[[6]],
                    plot_test[[7]], plot_test[[8]], plot_test[[9]],
                    nrow = 2),
       filename = paste0("./cdot/img/Test3bCDOT.png"),
       width = 16, height = 8)

# Correlated Test 3b Plot

for (i in 1:length(RR)) {
  df = subset(Test, h >= cut_offs_corr[i])
  title = paste0("BR:", RR[i], "%, M|h in [",
                 round(cut_offs_corr[i], 2), ",", 1, "]")

  plot_test[[i]] = ggplot(df, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(aes(y = ..density..),
                   alpha = 0.5, bins = 50, position = "identity") +
    lims(x = c(200, 1500)) +
    ggtitle(title) +
    scale_color_manual(values = c("3b" = "seagreen3",
                                  "Signal" = "blue")) +
    scale_fill_manual(values = c("3b" = "seagreen3",
                                 "Signal" = "blue")) +
    ylab("Density") +
    xlab("Mass") +
    my_theme +
    theme(legend.position = "none")
}

ggsave(grid.arrange(plot_test[[1]], plot_test[[2]], plot_test[[3]],
                    plot_test[[4]], plot_test[[5]], plot_test[[6]],
                    plot_test[[7]], plot_test[[8]], plot_test[[9]],
                    nrow = 2),
       filename = paste0("./cdot/img/Test3bcorr.png"),
       width = 16, height = 8)

# Test 4b data

plot_test = list(length = length(RR))
Test = rbind(Background.Test.4b, Signal.Test)
Test$Class = factor(Test$Class, levels = c("Signal", "4b"))

for (i in 1:length(RR)) {
  df = subset(Test, Trans_h >= cut_offs_CDOT[i])
  title = paste0("BR:", RR[i], "%, M|T(h) in [",
                 round(cut_offs_CDOT[i], 2), ",", 1, "]")

  plot_test[[i]] = ggplot(df, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(aes(y = ..density..),
                   alpha = 0.5, bins = 50, position = "identity") +
    lims(x = c(200, 1500)) +
    ggtitle(title) +
    scale_color_manual(values = c("4b" = "seagreen3",
                                  "Signal" = "blue")) +
    scale_fill_manual(values = c("4b" = "seagreen3",
                                 "Signal" = "blue")) +
    ylab("Density") +
    xlab("Mass") +
    my_theme +
    theme(legend.position = "none")
}

ggsave(grid.arrange(plot_test[[1]], plot_test[[2]], plot_test[[3]],
                    plot_test[[4]], plot_test[[5]], plot_test[[6]],
                    plot_test[[7]], plot_test[[8]], plot_test[[9]],
                    nrow = 2),
       filename = paste0("Test4bCDOT.png"),
       width = 16, height = 8)

# Correlated Test 4b Plot

for (i in 1:length(RR)) {
  df = subset(Test, h >= cut_offs_corr[i])
  title = paste0("BR:", RR[i], "%, M|h in [",
                 round(cut_offs_corr[i], 2), ",", 1, "]")

  plot_test[[i]] = ggplot(df, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(aes(y = ..density..),
                   alpha = 0.5, bins = 50, position = "identity") +
    lims(x = c(200, 1500)) +
    ggtitle(title) +
    scale_color_manual(values = c("4b" = "seagreen3",
                                  "Signal" = "blue")) +
    scale_fill_manual(values = c("4b" = "seagreen3",
                                 "Signal" = "blue")) +
    ylab("Density") +
    xlab("Mass") +
    my_theme +
    theme(legend.position = "none")
}

ggsave(grid.arrange(plot_test[[1]], plot_test[[2]], plot_test[[3]],
                    plot_test[[4]], plot_test[[5]], plot_test[[6]],
                    plot_test[[7]], plot_test[[8]], plot_test[[9]],
                    nrow = 2),
       filename = paste0("./cdot/img/Test4bcorr.png"),
       width = 16, height = 8)


# Test Data - Signal Enrichment Plots

plot_test = list(length = length(RR))
Test$Class = factor(Test$Class, levels = c("4b", "Signal"))

for (i in 1:length(RR)) {
  df = subset(Test, Trans_h >= cut_offs_CDOT[i])
  title = paste0("BR:", RR[i], "%, M|T(h) in [",
                 round(cut_offs_CDOT[i], 2), ",", 1, "]")

  plot_test[[i]] = ggplot(df, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(alpha = 0.5, bins = 50, position = "identity") +
    ggtitle(title) +
    scale_color_manual(values = c("4b" = "seagreen3",
                                  "Signal" = "blue")) +
    scale_fill_manual(values = c("4b" = "seagreen3",
                                 "Signal" = "blue")) +
    ylab("Density") +
    xlab("Mass") +
    my_theme +
    theme(legend.position = "none")
}

ggsave(grid.arrange(plot_test[[1]], plot_test[[2]], plot_test[[3]],
                    plot_test[[4]], plot_test[[5]], plot_test[[6]],
                    plot_test[[7]], plot_test[[8]], plot_test[[9]],
                    nrow = 2),
       filename = paste0("./cdot/img/Test4bCDOTEnriched.png"),
       width = 16, height = 8)

# Correlated Validation Plot

for (i in 1:length(RR)) {
  df = subset(Test, h >= cut_offs_corr[i])
  title = paste0("BR:", RR[i], "%, M|h in [",
                 round(cut_offs_corr[i], 2), ",", 1, "]")

  plot_test[[i]] = ggplot(df, aes(x = m4j, fill = Class, colour = Class)) +
    geom_histogram(alpha = 0.5, bins = 50, position = "identity") +
    ggtitle(title) +
    scale_color_manual(values = c("4b" = "seagreen3",
                                  "Signal" = "blue")) +
    scale_fill_manual(values = c("4b" = "seagreen3",
                                 "Signal" = "blue")) +
    ylab("Density") +
    xlab("Mass") +
    my_theme +
    theme(legend.position = "none")
}

ggsave(grid.arrange(plot_test[[1]], plot_test[[2]], plot_test[[3]],
                    plot_test[[4]], plot_test[[5]], plot_test[[6]],
                    plot_test[[7]], plot_test[[8]], plot_test[[9]],
                    nrow = 2),
       filename = paste0("./cdot/img/Test4bcorrEnriched.png"),
       width = 16, height = 8)



