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
        # sigma <- mclapply(1:cl, FUN = function(x, d){
        #     clusterGeneration::genPositiveDefMat(dim = d, covMethod = "unifcorrmat", rangeVar = c(25, 40))$Sigma
        # }, d = d, mc.cores = 6)
    }
    
    if(d==1){
        rnorm.gen <- rnorm
    }else{
        rnorm.gen <- MASS::mvrnorm
    }
    
    temp <- c()
    for(i in 1:cl){
        component <- rnorm.gen(mixing.n[i], mean[[i]], sigma[[i]])
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

writing.path <- "/media/syl/Ergo/Uczelnia/Projekty/PhD_unsupervised_comparisons/Data/data_raw/simulated_multiVar_normal_MASS/"
file.remove(list.files(paste0(writing.path, "meta_data"), full.names = TRUE))
file.remove(list.files(writing.path, full.names = TRUE))

if(!dir.exists(paste0(writing.path, "meta_data"))) dir.create(paste0(writing.path, "meta_data"))

clusters <- rep(2:6, each = 10)
dims <- c(seq(10, 100, by = 20), seq(200, 1000, by = 200))
n <- 1500
datatype <- expand.grid(n, clusters, dims, stringsAsFactors = F)
datatype$Var1 <- replicate(length(clusters), runif(10, min = 500, max = 1500) |> round()) |> as.vector()
#tapply(datatype$Var1, INDEX = datatype$Var3, FUN = sum)

for(i in 1:nrow(datatype)){
    print(i)
    uniqid <- paste0(sample(letters, 5), collapse = "")
    out <- normal.mixture.generator(n = datatype[i,1], cl = datatype[i,2], d = datatype[i,3])
    data.table::fwrite(x = out$data, file = paste0(writing.path, datatype[i,2],"_d", datatype[i,3], "_sim_multiVar_normal_", uniqid, ".dt"), col.names = TRUE)
    readr::write_rds(out[-6], file = paste0(writing.path, "meta_data/", "meta_", datatype[i,2], "_sim_multiVar_normal_", uniqid))
}

