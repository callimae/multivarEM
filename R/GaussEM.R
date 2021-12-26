# GaussEM version 0.5
apply.by.cluster <- function(empirical.data, clusters, MARGIN, fun, ..., mc.cores = parallel::detectCores()-1){
    k.clusters <- 1:max(clusters)
        parallel::mclapply(k.clusters, function(k, empirical.data, clusters, MARGIN, fun, ...){
            inx <- which(clusters == k)
            out <- apply(empirical.data[inx,], MARGIN = MARGIN, FUN = fun, ...)
        }, empirical.data = empirical.data, clusters = clusters, MARGIN = MARGIN, fun = fun, ...,
        mc.cores = 1)
}

gaussEM_ini <- function(empirical.data, k){
    clusters <- rep(0, times = nrow(empirical.data))
    for(k.el in 1:k){
        clusters[sample(nrow(empirical.data), size = 10)] <- k.el
        }
    clusters <- (kmeans(x = empirical.data, centers = k))$cluster
    means.list <- apply.by.cluster(empirical.data = empirical.data, clusters = clusters, MARGIN = 2, fun = mean)
    vars.list <- apply.by.cluster(empirical.data = empirical.data, clusters = clusters, MARGIN = 2, fun = var)
    alpha <- runif(k, min = 0.05); alpha <- alpha/sum(alpha)
    out <- list("means" = means.list,
                "vars" = vars.list,
                "alphas" = alpha)
    return(out)
}

gauss_MAX <- function(ll_prior, empirical.data, varmin, k){
    ll_prior.df <- do.call(cbind, ll_prior)
    #denum <- rowLogSumExps(ll_prior.df) # Which one is better? Here we are standardizing as usual.
    denum <- apply(ll_prior.df, 1, max) # Which one is better? Here we take maximum value
    post <- exp(ll_prior.df - denum)
    post_plus <- post/rowSums(post) # allows us to avoid standardization of alphas, means, sds
    post_plus_sum <- colSums(post_plus)

    alphas <- post_plus_sum/nrow(post)
    if (any(alphas<0.001)){alphas[alphas < 0.001] <- 0.01}

    means <- crossprod(post_plus, as.matrix(empirical.data))/post_plus_sum
    means <- lapply(1:k, function(k, means) means[k,], means)
    vars <- lapply(1:k, function(k, means, empirical.data, post_plus){
        xvar <- tcrossprod(post_plus[,k], ((t(empirical.data) - means[[k]])^2))
        xvar <- c(xvar/sum(post_plus[,k]))
        xvar[xvar < varmin[[k]]] <- varmin[[k]][xvar < varmin[[k]]]
        return(xvar)
    }, means = means,
    empirical.data = empirical.data,
    post_plus = post_plus)
    out <- list("posterior" = post,
                "ll_prior" = ll_prior.df,
                "means" = means,
                "vars" = vars,
                "alphas" = alphas)
}

gaussEM_change <- function(params0, params){
   rbind_xy_diff  <- function(x, y){
        sum(do.call(rbind, x) - do.call(rbind, y))
    }
    means <- rbind_xy_diff(x = params0$means, y = params$means)
    vars <- rbind_xy_diff(x = params0$vars, y = params$vars)
    alphas <-  params0$alphas - params$alphas
    out <- abs(sum(c(means, vars, alphas)))
}

GaussEM <- function(empirical.data, k){
    library(matrixStats)
    params <- gaussEM_ini(empirical.data = empirical.data, k = k)
    varmin <- lapply(params$vars, function(x) x * 1e-4)
    ones1N <- rep(1, times = nrow(empirical.data))
    change <- 1
    cat("*** GaussEM ***\n")
    cat("Number of components: ", crayon::bold(k), "\n")
while (change > 1e-5){
    params0 <- params
    ll_prior <- parallel::mclapply(1:k, function(k, empirical.data, means, vars, alphas, ones1N){
                    nvar = ncol(empirical.data)
                    #xi_xmean_var <- rowSums(t(t((empirical.data)-means[[k]])^2/vars[[k]]))
                    #xi_xmean_var <- t(t((empirical.data)-means[[k]])^2/vars[[k]]) |> rowSums()
                     # out <- log(alphas[[k]]) - lvar_lpi - xi_xmean_var
                     # xi_xmean2 <- (t(empirical.data)-means[[k]])^2
                     # xi_xmean2_var <- colSums(xi_xmean2/vars[[k]])/2
                     # lvar_lpi <- sum(log((vars[[k]])))/2 - (nvar/2)*log(2*pi)
                     # out <- log(alphas[[k]]) - lvar_lpi - xi_xmean2_var

                    a <- log(alphas[[k]])*ones1N # alpha parameter multiplicated by number of rows
                    b <- colSums((t(empirical.data) - means[[k]])^2/(vars[[k]]), na.rm = T)/2
                    c <- sum(log((vars[[k]])))*ones1N/2 - (nvar/2)*log(2*pi)
                    #browser()
                    out <- a-b-c
                    return(out)
                }, empirical.data = empirical.data, ones1N = ones1N, mc.cores = 4,
                means = params$means,
            vars = params$vars,
        alphas = params$alphas)
    params <- gauss_MAX(ll_prior, empirical.data, varmin = varmin, k = k)
    # cat("Means", do.call(rbind, params$means), "\n")
    # cat("vars", do.call(rbind, params$vars), "\n")
    # cat("alphas",  params$alphas)
    if(any(is.nan(do.call(rbind, params$means)))) {browser()}
    change <- gaussEM_change(params0 = params0, params = params)
    cat("Change:", change, "\r")

}
return(apply(params$ll_prior, 1, which.max))
}
