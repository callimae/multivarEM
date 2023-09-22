gaussEM_ini <- function(xdata, k){
    clusters <- params_ini(xdata = xdata, k = k, type = "random")
    means.list <- apply_by_cluster(xdata = xdata, fun = matrixStats::colMeans2, clusters = clusters)
    vars.list <- (abs(means.list)) + 1e-4
    vars.list[is.na(vars.list)] <- abs(means.list[is.na(vars.list)])
    vars.list[vars.list <= 0] <- runif(sum(vars.list <= 0), min = 1e-6, max = max(vars.list))
    varmin <- abs(vars.list)
    alpha <- runif(k, min = 0.05); alpha <- alpha/sum(alpha)
    out <- list("means" = means.list,
                "vars" = vars.list,
                "varmin" = varmin,
                "alphas" = alpha)
    return(out)
}

gauss_MAX <- function(ll_prior, xdata, txdata, varmin, k, vars, cenv){
    denum <- Rfast::colMaxs(ll_prior, value = TRUE)
    post <- ll_prior - denum
    post_plus <- exp(post - matrixStats::rowLogSumExps(post))#+1e-311 # allows us to avoid standardization of alphas, means, sds
    post_plus_sum <- Rfast::colsums(post_plus)
    means2 <- cenv$params$means
    alphas2 <- cenv$params$alphas
    vars2 <- cenv$params$vars
    alphas <- post_plus_sum/nrow(post)
    means <- crossprod(post_plus, xdata)/post_plus_sum
    for(i in seq_len(k)){
        xvar <- tcrossprod(post_plus[,i], ((txdata - means[i,])^2))
        vars[i,] <- c(xvar/sum(post_plus[,i]))
        vars[i,][vars[i,]<=0] <- varmin[i,][vars[i,]<=0]
    }
    varmin <- vars
    mean = (means + means2)/2
    vars = (vars + vars2)/2
    alphas = (alphas + alphas2)/2
    ll <- sum(denum) + matrixStats::logSumExp(post)
    out <- list("posterior" = post, "ll_prior" = ll_prior,
                "ll" = ll, "varmin" = varmin,
                "means" = means, "vars" = vars, "alphas" = alphas)
}

GaussEM <- function(xdata, k, em.itr = 1500, tol = 1e-8, start_ini = 10, start_ini_cores = 1, obs.names = NULL){
    if(!any(class(xdata) == "matrix")){
        xdata <- as.matrix(xdata)
        mode(xdata) <- "numeric"
    }
    if(anyNA(xdata)){
        errorCondition("Data contains NA values. Remove or impute them.")
    }

    cenv <- environment()
    llmvnorm <- function(xdata, vmeans, vvars, valph, nvar){
        xi_xmean <- (xdata-vmeans)^2
        xi_xmean_var <- Rfast::colsums(xi_xmean/vvars)/2
        lvar_lpi <- sum(log((vvars)))/2 - (nvar/2)*log(2*pi)
        return(log(valph) - lvar_lpi - xi_xmean_var)
    }

    cat(crayon::green("*** GaussEM ***\n"))
    cat("Components: ", crayon::bold(k), "\n")

    if(nrow(xdata)==1){
        warning("Data has only one observation")
        params <- list("cluster" = 1)
        return(params)
    }

    # Define function for difference in paramters from internal params_change
    par_change <- params_change(type = "normal")

    if(is.null(obs.names)){
        obs.names <- 1:nrow(xdata)
    }
    # Initialization
        txdata <- t.default(xdata)
        bic.counter <- 0
        dimdata <- dim(xdata)
        .delta = 100; itr = 0; ll = c(-Inf)
        nvar <- ncol(xdata)
        ll_prior <- matrix(nrow = nrow(xdata), ncol = k)

    ll_try <- parallel::mclapply(1:start_ini, function(x){
        params <- gaussEM_ini(xdata = xdata, k = k)
        ll_prior <- parallel::mclapply(seq_len(k), function(x, xdata, vmeans, vvars, valph, nvar){
            llmvnorm(xdata, vmeans[x,], vvars[x,], valph[x], nvar)
        }, xdata = txdata, vmeans = params$means, vvars = params$vars,
        valph = params$alphas,nvar = nvar, mc.preschedule = T, mc.cores = 1)
        ll_prior <- do.call(cbind, ll_prior)
        params$ini_ll <- exp(ll_prior) |> sum() |> log()
        return(params)
    }, mc.cores = start_ini_cores)
    params <- ll_try[[which.max(sapply(ll_try, function(x) x$ini_ll))]]; rm(ll_try)

    while (.delta >= tol & itr <= em.itr){
        params0 <- params
        ll_prior <- parallel::mclapply(seq_len(k), function(x, xdata, vmeans, vvars, valph, nvar){
            llmvnorm(xdata, vmeans[x,], vvars[x,], valph[x], nvar)
        }, xdata = txdata, vmeans = params$means, vvars = params$vars,
        valph = params$alphas,nvar = nvar, mc.preschedule = T, mc.cores = 1)
        ll_prior <- do.call(cbind, ll_prior)
        params <- gauss_MAX(ll_prior, xdata = xdata, txdata = txdata,
                            varmin = params$varmin, k = k,
                            vars = params$vars, cenv = cenv)
        #if(abs(ll[length(ll)]-params$ll) < 1e-8){break}
        ll <- c(ll, params$ll)
        if(.delta == par_change(params0 = params0, params = params)){break}
        .delta <- par_change(params0 = params0, params = params)
        itr = itr + 1

        cat("Change:", .delta, ": ", itr, "\r")
    }
        params$cluster <- apply(params$ll_prior, 1, which.max)
        bic <- log(prod(dimdata))*((k-1)+2*k*dimdata[2])-2*ll
        names(params$cluster) <- obs.names
        params$varmin <- NULL
        params$bic <- bic
        params$ll <- ll
        return(params)
}
