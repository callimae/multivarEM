library(MultiRNG)
library(parallel)
normal.mixture.generator <- function(n = NULL, cl = NULL, d = NULL){
    library(MASS)
    if (is.null(n)) n = 1000
    if (is.null(cl)) cl = 2
    if (is.null(d)) d = 1

    mixing.proportions <- runif(n = cl, min = 0.1)
    mixing.proportions.standarized <- mixing.proportions/sum(mixing.proportions)
    mixing.n <- round(mixing.proportions.standarized * n, 0)

    # simple generation - might be changed to more sophisticated functions
    #mean <- replicate(cl, runif(n=d, 3, 6), simplify=FALSE)
    mean <- replicate(cl, runif(n=d, runif(1,0,3), runif(1,3.1,6)), simplify=FALSE)

    if(d==1){
        sigma <- replicate(cl,  runif(1, min = 3, max = 6))
    }else{
        sigma <- replicate(cl, if(d==1) runif(1, min = 3, max = 6) else diag(runif(n = d, min = 16, max = 38)), simplify=FALSE)
    }

        if(d==1){
            rnorm.gen <- rnorm
        }else{
            rnorm.gen <- MultiRNG::draw.d.variate.normal
        }

        temp <- c()
        for(i in 1:cl){
            component <- rnorm.gen(mixing.n[i], mean[[i]], sigma[[i]], d = d)
            #component <- rnorm.gen(mixing.n[i], mean[[i]], sigma[[i]])
            if(d==1){
                temp <- c(temp, component)
            }else{
                temp <- rbind(temp, component)
            }
        }

    temp <- data.table::data.table(temp)
    temp$cluster <- rep(letters[1:cl], times = mixing.n)

    out <- list("mixing_proportions" = mixing.proportions.standarized,
                "mixing.n" = mixing.n,
                "mean" = mean,
                "sigma" = sigma,
                "clusters" = cl,
                "data" = temp)

    return(out)
}
