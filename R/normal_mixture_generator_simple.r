
normal.mixture.generator <- function(n = NULL, cl = NULL, d = NULL){
    library(MASS)
    if (is.null(n)) n = 1000
    if (is.null(cl)) cl = 2
    if (is.null(d)) d = 3
    
    mix.p <- runif(cl, min = 0.1)
    mix.p <- mix.p/sum(mix.p)
    
    # zadajemy parametry micltury Gaussowskiej
    miv <- t(replicate(cl, runif(d)))-0.5
    sigv <- t(replicate(cl, runif(d)))
    
    # tworzymy listy
    miv.2lt <- lapply(1:nrow(miv), function(i, miv) {
        miv.lt <- miv[i,]
        lapply(seq_len(length(miv.lt)), function(j, miv.lt){
            miv.lt[i]
        }, miv.lt = miv.lt)
    }, miv = miv)
    
    sigv.2lt <- lapply(1:nrow(sigv), function(i, sigv) {
        sigv.lt <- sigv[i,]
        lapply(seq_len(length(sigv.lt)), function(j, sigv.lt){
            sigv.lt[i]
        }, sigv.lt = sigv.lt)
    }, sigv = sigv)
    
    mix.p.n <- round(mix.p * n)
    mix.p.n.lt <- lapply(1:length(mix.p.n), function(i, mix.p.n) mix.p.n[i], mix.p.n = mix.p.n)
    # generujemy liczby probek z poszczegolnych skladowych
    
    mix.comps <- mapply(function(miv.2lt, sigv.2lt, mix.p.n.lt){
        # We are enter two element list, where each element have other list inside (apart from mix.p.lt)
        out <- parallel::mcmapply(function(miv.2lt, sigv.2lt, mix.p.n.lt){
            out <- rnorm(mix.p.n.lt, mean = miv.2lt, sd = sigv.2lt) #at this deep, miv.2lt and others have only one single value. 
        }, miv.2lt = miv.2lt, sigv.2lt = sigv.2lt, mix.p.n.lt = mix.p.n.lt, mc.cores = 4)
    }, miv.2lt = miv.2lt, sigv.2lt, mix.p.n.lt)
    out <- do.call(rbind, mix.comps) |> data.table::as.data.table()
    out$cluster <- rep(letters[1:cl], times = mix.p.n)
    
    
    out <- list("mixing_proportions" = mix.p,
                "mixing.n" = mix.p.n,
                "mean" = miv,
                "sigma" = sigv,
                "clusters" = cl,
                "data" = out)
    
    return(out)
}

writing.path <- "/media/syl/Ergo/Uczelnia/Projekty/PhD_unsupervised_comparisons/Data/data_raw/simulated_multiVar_normal_MultiRNG//"
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
    data.table::fwrite(x = out$data, file = paste0(writing.path, datatype[i,2], "_d", datatype[i,3], "_sim_multiVar_normal_", uniqid, ".dt"), col.names = TRUE)
    readr::write_rds(out[-6], file = paste0(writing.path, "meta_data/", "meta_", datatype[i,2], "_sim_multiVar_normal_", uniqid))
}

