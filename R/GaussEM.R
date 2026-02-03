params_change <- function(type = c("poiss", "normal", "multinomial")){
  type = match.arg(type)
  switch (type,
          "normal"  = {
            par_change <- function(params0, params){
              means <- sum(params0$means -params$means)
              vars <- sum(params0$vars - params$vars)
              alphas <- sum(params0$alphas - params$alphas)
              out <- abs(sum(c(means, vars, alphas)))
              return(out)
            }
          },
          "multinomial"  = {
            par_change <- function(params0, params){
              return(out)
            }
          }
  )
  return(par_change)
}

mstep <- function(ll_prior, xdata, varmin, k, vars, cenv){
    denum <- Rfast::colMaxs(ll_prior, value = TRUE)
    post <- ll_prior - denum
    post_plus <- exp(post - rowLogSumExps(post))#+1e-311 # allows us to avoid standardization of alphas, means, sds
    post_plus_sum <- Rfast::colsums(post_plus)
    post_plus_sum <- post_plus_sum + 1/sum(post_plus_sum)
    alphas <- post_plus_sum/nrow(post)
    #means <- crossprod(post_plus, xdata)/post_plus_sum
    means <- compute_means_simple(post_plus, xdata, post_plus_sum)
    vars <- calculate_vars(post_plus,xdata,means,varmin,k)
    varmin <- vars
    ll <- sum(denum) + matrixStats::logSumExp(post)
    out <- list("posterior" = post, "ll_prior" = ll_prior,
                "ll" = ll, "varmin" = varmin,
                "means" = means, "vars" = vars, "alphas" = alphas)
}

estep <- function(k, xdata, params, .nvar = nvar){
  ll_prior <- lapply(seq_len(k), function(x, xdata, vmeans, vvars, valph, nvar){
    llmvnormParallelOptimized(xdata, vmeans[x,], vvars[x,], valph[x], nvar, grain_size = 50)
  }, xdata = xdata, vmeans = params$means, vvars = params$vars,
  valph = params$alphas, nvar = .nvar)
  ll_prior <- do.call(cbind, ll_prior)
  return(ll_prior=ll_prior)
}


GaussEM <- function(xdata, k, em.itr = 1500, tol = 1e-8, cores = 1,
                    start_ini = 25, start_ini_cores = 1, obs.names = NULL, 
                    ini = "random", kmeans_iter = 25, kmeans_init = 3){

    if(!any(class(xdata) == "matrix")){
        xdata <- as.matrix(xdata)
        mode(xdata) <- "numeric"
    }
    if(anyNA(xdata)){
        errorCondition("Data contains NA values. Remove or impute them.")
    }

    cenv <- environment()
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
        bic.counter <- 0
        dimdata <- dim(xdata)
        .delta = 100; itr = 0; ll = c(-Inf)
        nvar <- ncol(xdata)
        ll_prior <- matrix(nrow = nrow(xdata), ncol = k)
        if(ini != "random") start_ini = 1
    ll_try <- lapply(1:start_ini, function(x){
        params <- gaussEM_ini_cpp_parallel(xdata = xdata, k = k, ini = ini, 
                                          kmeans_iter = kmeans_iter, 
                                          kmeans_init = kmeans_init)
        ll_prior <- estep(k, xdata, params, .nvar = nvar)
        params$ini_ll <- exp(ll_prior) |> sum() |> log()
        return(params)
    })
    params <- ll_try[[which.max(sapply(ll_try, function(x) x$ini_ll))]]; rm(ll_try)

    while (.delta >= tol & itr <= em.itr){
        params0 <- params

        ll_prior <- estep(k, xdata, params, .nvar = nvar)
        params <- mstep(ll_prior, xdata = xdata, varmin = params$varmin, k = k,
                            vars = params$vars, cenv = cenv)
        #if(abs(ll[length(ll)]-params$ll) < 1e-8){break}
        ll <- c(ll, params$ll)
        if(.delta == par_change(params0 = params0, params = params)){break}
        .delta <- par_change(params0 = params0, params = params)
        itr = itr + 1

        cat("Change:", .delta, ": ", itr, "\r")
    }
        params$cluster <- apply(ll_prior, 1, which.max)
        bic <- log(prod(dimdata))*((k-1)+2*k*dimdata[2])-2*ll
        names(params$cluster) <- obs.names
        params$varmin <- NULL
        params$bic <- bic
        params$ll <- ll
        params$iter <- itr
        return(params)
}