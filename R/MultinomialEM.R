timer <- local({
    timer_env <- new.env()
    list(
        start = function() {
            invisible(assign(envir = timer_env, x = "start_time", value = Sys.time()))
        },
        stop = function(){
            invisible(assign(envir = timer_env, x = "stop_time", value = Sys.time()))
            return(timer_env$stop_time - timer_env$start_time)
        }
    )
})

multinomial_mstep <- function(pdk, xdata, cenv){
    alphas <- Rfast::colsums(pdk)/nrow(xdata)
    prob.mat <- abs(crossprod(pdk, xdata))
    rS <- rowSums(prob.mat)
    rS[rS==0] <- min(rS[rS!=0])
    prob.mat.st <- prob.mat/rS
    zero_count <- sum(prob.mat.st == 0)
    if (zero_count > 0) {
        prob.mat.st[prob.mat.st==0] <- runif(n = zero_count, min = 1e-311, max = 1e-200)
    }
    prob.mat.st <- (prob.mat.st + cenv$params$probs)/2
    params <- list("probs" = prob.mat.st,
                   "alphas" = alphas)
    return(params)
}
multinomial_estep <- function(xdata, params){
    log_alphas <-  log(params$alphas)
    post_list_check <- tcrossprod(xdata, log(params$probs))
    for(i in 1:NCOL(post_list_check)){
        post_list_check[,i] <- post_list_check[,i] + log_alphas[i]
    }
    pdk <- exp(post_list_check - matrixStats::rowLogSumExps(post_list_check))
    return(list(post_list_check = post_list_check, pdk = pdk))
}

multinomEM <- function(xdata, k, ini_type, itrs = 1000){
    cenv <- environment()
    dims <- dim(xdata)
    timer$start()
    rep <- 0
    repeat{
        if(ini_type == "kmeans" || ini_type == "kmeans++"){
            start = 1
        }else{
            start = 25
        }
        ll_try <- parallel::mclapply(1:start, function(x){
            params <- improved_multiEM_ini(xdata = xdata, k = k, ini_type = ini_type)
            post_list_check <- tcrossprod(xdata, log(params$probs))
            for(i in 1:NCOL(post_list_check)){
                post_list_check[,i] <- post_list_check[,i] + log(params$alphas)[i]
            }
            params$ini_ll <- logSumExp(post_list_check)
            return(params)
        }, mc.cores = 1)

        params <- ll_try[[which.max(sapply(ll_try, function(x) x$ini_ll))]]; rm(ll_try)

        .delta <- 100; log.lik <- c(); BICvec <- c(); itr <- 0
        cat("*** MultinomEM ***\n")
        cat("Components: ", crayon::bold(k), "\n")
        
        while (.delta > 1e-6 & itr != itrs){
            itr <- itr + 1
            params0 <- params
            estep <- multinomial_estep(xdata = xdata, params = params)
            params <- multinomial_mstep(pdk = estep$pdk, xdata = xdata, cenv)
            loglik <- sum(estep$post_list_check*estep$pdk, na.rm = T)
            log.lik <- c(log.lik, loglik)
            #.BIC <- (dims[2]*k+k) * log(dims[1]*dims[2]) - 2*loglik; #print(.BIC)
	    .BIC <- (k * dims[2] + k - 1) * log(dims[1]) - 2 * loglik
            BICvec <- c(BICvec, .BIC) #dims: n (no obs.); dims[2] + cl: number of parameters
            .delta <- sum(abs(params0$alphas - params$alphas))+sum(params0$probs - params$probs)

            cat(crayon::blue(.delta),": ", itr, "\r")
        }

        tm <- print(timer$stop())
        if(length(unique(apply(estep$pdk, 1, which.max))) == k){break}
        if(rep == 1){break}; rep <- rep + 1
    }
    return(list(pdk = estep$pdk, params = params, clusters = apply(estep$pdk, 1, which.max), tm = tm, itr = itr, BIC = BICvec))
}
