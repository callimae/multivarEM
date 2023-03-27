multiEM_ini <- function(xdata, k, ini.type){
    dims <- dim(xdata)
    switch(ini.type,
           "random" = {
               prob.mat <- t(replicate(k, rpois(dims[2], lambda = dims[2])))
               prob.mat.st <- prob.mat/rowSums(prob.mat)
            },
           "kmeans" = {
               out <- kmeans(xdata, centers = k, iter.max = 1)
               prob.mat <- abs(out$centers)
               prob.mat.st <- prob.mat / rowSums(prob.mat)
            },
           "hclust" = {
               data.samples <- round(dim(xdata)[1] * .55) #0.25 is used to shorten the time of the algorithm
               inx.samples <- sample(nrow(xdata), size = data.samples)
               data.sampled <- xdata[inx.samples,]
               d <- dist(data.sampled)
               hc <- hclust(d, method = "ward.D2")
               out <- cutree(hc, k = k)
               #browser()
               data.sampled.list <- split(data.sampled, f = 1:length(out))
               data.sampled.clusters <- lapply(split(data.sampled.list, f = out),
                                               FUN = function(x) {x <- do.call(rbind, x); x <- colSums(x)+1; x/sum(x)})
               prob.mat.st <- as.matrix(do.call(rbind, data.sampled.clusters))
           })

    alphas <- runif(n = k, min = 0.1)
    alphas <- alphas/sum(alphas)
    out <- list("probs" = prob.mat.st,
                "alphas" = alphas)
    return(out)
}

multinomEM <- function(xdata, k, ini.type, itrs = 1000){

    if(anyNA(xdata)){
        xdata <- xdata[,-which(is.na(colSums(xdata)))]
    }
    cenv <- environment()
    if(dim(xdata) |> prod() > 6e+4){
        prodcross <- Rfast::Crossprod
    }else{
        prodcross <- crossprod
    }
    #par_change <- params_change(type = "multinomial")
    if(!any(class(xdata) == "matrix")){
        xdata <- as.matrix(xdata)
        mode(xdata) <- "numeric"
    }
    dims <- dim(xdata)
    a <- Sys.time()
    multi_MAX <- function(pdk, xdata, cenv){
        alphas <- Rfast::colsums(pdk)/nrow(xdata)
        prob.mat <- abs(prodcross(pdk, xdata))
        # prob.mat <- exp(prob.mat - matrixStats::rowLogSumExps(prob.mat))
        #if(any(prob.mat==0)){browser()}
        rS <- rowSums(prob.mat)
        rS[rS==0] <- min(rS[rS!=0])
        prob.mat.st <- prob.mat/rS
        prob.mat.st[prob.mat.st==0] <- runif(n = length(sum(prob.mat.st==0)), min = 1e-311, 1e-200)
        prob.mat.st <- (prob.mat.st + cenv$params$probs)/2
        params <- list("probs" = prob.mat.st,
                       "alphas" = alphas)
        return(params)
    }
    rep <- 0
    repeat{
        #params <- multiEM_ini(xdata = xdata, k = k, ini.type = ini.type)
        ll_try <- bettermc::mclapply(1:100, function(x){
            params <- multiEM_ini(xdata = xdata, k = k, ini.type = "random")
            post.list.check <- tcrossprod(xdata, log(params$probs)) + log(params$alphas)
            params$ini_ll <- logSumExp(post.list.check)
            return(params)
        }, mc.cores = 15)

        params <- ll_try[[which.max(sapply(ll_try, function(x) x$ini_ll))]]; rm(ll_try)

        .delta <- 100; log.lik <- c(); BICvec <- c(); itr <- 0
        cat("*** MultinomEM ***\n")
        cat("Components: ", crayon::bold(k), "\n")
        #
        while (.delta > 1e-6 & itr != itrs){
            itr <- itr + 1
            params0 <- params
            post.list.check <- Rfast::Tcrossprod(xdata, log(params$probs)) + log(params$alphas)
            pdk <- exp(post.list.check - matrixStats::rowLogSumExps(post.list.check))
            # MAXIMIZATION
            params <- multi_MAX(pdk = pdk, xdata = xdata, cenv)
            # convergance
            loglik <- sum(post.list.check*pdk, na.rm = T)+sum(t(pdk)*log(params$alphas), na.rm = TRUE)
            log.lik <- c(log.lik, loglik)
            .BIC <- (dims[2]*k+k) * log(dims[1]*dims[2]) - 2*loglik; #print(.BIC)
            BICvec <- c(BICvec, .BIC) #dims: n (no obs.); dims[2] + cl: number of parameters
            .delta <- sum(abs(params0$alphas - params$alphas))+sum(params0$probs - params$probs)
            #cat(crayon::blue(.delta), "\r")
            cat(crayon::blue(.delta),": ", itr, "\r")
        }
        b <- Sys.time()
        tm <- print(b - a)
        if(length(unique(apply(pdk, 1, which.max))) == k){break}
        if(rep == 1){break}; rep <- rep + 1
    }
    return(list(pdk = pdk, params = params, clusters = apply(pdk, 1, which.max), tm = tm, itr = itr, BIC = BICvec))
}
