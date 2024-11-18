source("./cdot/Requirements.R")
library(np)

conditional.scale.y <- function(x, y, fit.mu = NULL, fit.sigma = NULL) {
  if (is.null(fit.mu)) {
    fit.mu = smooth.spline(x = x, y = y, cv = TRUE)
  }
  predicted.mu = predict(fit.mu, x = x)$y
  eta = y - predicted.mu

  if (is.null(fit.sigma)) {
    fit.sigma = smooth.spline(x = x, y = eta^2, cv = TRUE)
  }
  predicted.sigmasq = predict(fit.sigma, x = x)$y
  e = eta / sqrt(predicted.sigmasq)
  trans.y = e
  return(list(trans.y = trans.y,
              fit.mu = fit.mu,
              fit.sigma = fit.sigma))
}


fit.CDOT <- function(x, y, scaled = TRUE,
                     logit.y.scale = TRUE, log.x.scale = TRUE,
                     splits = NULL, bws = NULL,
                     sub.sample = 3000, sub.sample.limit = 5000) {
  # Estimates Conditional Distribution for y|x.
  # scaled is for whether we want to scale y as (y - E[y|x])/sd(y|x).
  # logit.y.scale for whether to perform logit transformation on y.

  if (logit.y.scale) { trans.y = log(y / (1 - y)) }else { trans.y = y }
  if (log.x.scale) { trans.x = log(x) }else { trans.x = x }

  if (scaled) {
    scaled.fit = conditional.scale.y(x = trans.x, y = trans.y)
    trans.y = scaled.fit$trans.y
  }else {
    scaled.fit = NULL
  }

  splits = c(0, splits, Inf)
  index = vector(mode = "list", length = length(splits) - 1)

  if (is.null(bws)) {

    bws.m = rep(0, length(splits) - 1)
    bws = vector(mode = "list", length = length(splits) - 1)

    if (length(splits) == 2) {
      if (length(x) <= sub.sample.limit) {
        bws[[1]] = npcdistbw(xdat = x, ydat = trans.y)
      }else {
        index.sub = sample(1:length(trans.x), size = sub.sample)
        bws[[1]] = npcdistbw(xdat = x[index.sub], ydat = trans.y[index.sub])
      }
      index[[1]] = c(1:length(trans.x))
    }else {
      if (length(x) <= sub.sample.limit) {
        bws.h = npcdistbw(xdat = trans.x, ydat = trans.y)
        bws.h = bws.h$ybw
      }else {
        index.sub = sample(1:length(trans.x), size = sub.sample)
        bws.h = npcdistbw(xdat = trans.x[index.sub], ydat = trans.y[index.sub])
        bws.h = bws.h$ybw
      }

      for (i in 1:(length(splits) - 1)) {

        index[[i]] = which(x >= splits[i] & x <= splits[i + 1])

        if (length(index[[i]]) <= sub.sample.limit) {
          bws.temp = npcdistbw(xdat = trans.x[index[[i]]], ydat = trans.y[index[[i]]])
          bws.m[i] = bws.temp$xbw
        }else {
          index.sub = sample(index[[i]], size = sub.sample)
          bws.temp = npcdistbw(xdat = trans.x[index.sub], ydat = trans.y[index.sub])
          bws.m[i] = bws.temp$xbw
        }

        bws[[i]] = bws.temp

        bws[[i]]$ybw = bws.h


      }
    }

  }else {
    stop("Use predict.CDOT function!")
  }

  return(list(splits = splits, index = index, bws = bws,
              x = x, y = y, trans.x = trans.x, trans.y = trans.y,
              scaled.fit = scaled.fit))
}

predict.CDOT <- function(CDOT.model,
                         x, y,
                         scaled = TRUE,
                         logit.y.scale = TRUE, log.x.scale = TRUE,
                         splits = NULL) {

  if (logit.y.scale) { trans.y = log(y / (1 - y)) }else { trans.y = y }
  if (log.x.scale) { trans.x = log(x) }else { trans.x = x }

  if (scaled) {
    scaled.fit = conditional.scale.y(x = trans.x, y = trans.y,
                                     fit.mu = CDOT.model$scaled.fit$fit.mu,
                                     fit.sigma = CDOT.model$scaled.fit$fit.sigma)
    trans.y = scaled.fit$trans.y
  }

  splits = c(0, splits, Inf)
  index = vector(mode = "list", length = length(splits) - 1)
  CCDE = rep(1000, length(x))
  bws = CDOT.model$bws
  if (!is.null(bws)) {

    for (i in 1:(length(splits) - 1)) {

      index[[i]] = which(x >= splits[i] & x <= splits[i + 1])
      if (length(index[[i]]) > 0) {
        CCDE.fit = npcdist(bws[[i]],
                           txdat = CDOT.model$trans.x, tydat = CDOT.model$trans.y,
                           exdat = trans.x[index[[i]]], eydat = trans.y[index[[i]]])

        CCDE[index[[i]]] = CCDE.fit$condist
      }
    }

    Trans_h = quantile(CDOT.model$y, prob = CCDE)

  }else {
    stop("Use fit.CDOT function!")
  }

  return(list(splits = splits, index = index,
              CCDE = CCDE, Trans_h = Trans_h,
              trans.x = trans.x, trans.y = trans.y))
}
