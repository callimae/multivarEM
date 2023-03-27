apply_by_cluster <- function(xdata, fun, clusters, mc.cores = 1){
    out <- do.call(rbind, parallel::mclapply(sort(unique(clusters)), function(k, xdata){
    fun(xdata[which(clusters == k),])
    }, xdata = xdata, mc.cores = 1))
    return(out)
}

params_change <- function(type = c("poiss", "normal", "multinomial")){
    typ = match.arg(type)
    switch (type,
            "normal"  = {
                par_change <- function(params0, params){
                    means <- sum(params0$means -params$means)
                    vars <- sum(params$vars - params$vars)
                    alphas <-  params0$alphas - params$alphas
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

params_ini <- function(xdata, k, type = NULL, ...){
    sample_for_all <- function(k, xdata){
        out <- sample(k, size = nrow(xdata), replace = TRUE)
        if(length(unique(out))==k){
            return(out)
        }else{
            sample_for_all(k, xdata)
        }
    } 

    hc <- function(k, xdata, ...){
        args_list <- list(...)
            if(is.null(args_list$dist)){args_list$dist <- "euclidean"}
            if(is.null(args_list$link)){args_list$link <- "ward.D"}
        data.samples <- round(dim(xdata)[1] * .20) #0.20 is used to shorten the time of the algorithm
        inx.samples <- sample(nrow(xdata), size = data.samples)
        xdata.sampled <- data[inx.samples,]
        d <- dist(xdata.sampled, method = args_list$dist)
        hc <- hclust(d, method = args_list$link)
            return(cutree(hc, k = k))
    }
  
    if(is.null(type)){type = "random"}

    switch(type,
           "random" = {
              clusters <- sample_for_all(k = k, xdata = xdata)
               return(clusters)
           },
           "kmeans" = {
               clusters <- ClusterR::KMeans_rcpp(data = xdata, clusters = k)$clusters
               return(clusters)
           },
           "hc" = {
               clusters <- hc(k = k, xdata = xdata)
               return(clusters)
           })
}
