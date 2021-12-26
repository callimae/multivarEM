library(MultiRNG)
library(MASS)
library(parallel)
multinomial.mixture.generator <- function(n = NULL, cl = NULL, d = NULL){
    if (is.null(n)) n = 1000
    if (is.null(cl)) cl = 2
    if (is.null(d)) d = 1

mix.prop <- runif(n = cl, min = 0.1)
mix.prop.st <- mix.prop/sum(mix.prop)
mix.prop.n <- round(n * mix.prop.st)
thetas <- replicate(n = cl, rpois(n = d, lambda = round(sqrt(d))+1), simplify = F) # add plus 1 to avoid 0
thetas.st <- lapply(thetas, function(x) x/sum(x))
mix.components <- mcmapply(function(mix.prop.n, thetas.st, N){
    out <- draw.multinomial(mix.prop.n, d=N, theta=thetas.st, N=floor(N^1.2))
}, mix.prop.n = mix.prop.n, thetas.st = thetas.st, N=d, mc.cores = 1)

    temp <- do.call(rbind, mix.components) |> data.table::as.data.table()
    temp$cluster <- rep(letters[1:cl], times = mix.prop.n)

    out <- list("mixing_proportions" = mix.prop.st,
                "mixing.n" = mix.prop.n,
                "thetas" = thetas.st,
                "sizes" = 0.5*d,
                "clusters" = cl,
                "data" = temp)
    return(out)
}

writing.path <- "/media/syl/Ergo/Uczelnia/Projekty/PhD_unsupervised_comparisons/Data/data_raw/simulated_multinomial/"
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
    uniqid <- paste0(sample(letters, 3), collapse = "")
    out <- multinomial.mixture.generator(n = datatype[i,1], cl = datatype[i,2], d = datatype[i,3])
    data.table::fwrite(x = out$data, file = paste0(writing.path, datatype[i,2],"_d", datatype[i,3], "_simulationMultinom_", uniqid, ".dt"), col.names = TRUE)
    readr::write_rds(out[-6], file = paste0(writing.path, "meta_data/", "meta_", datatype[i,2], "_simulationMultinom_", uniqid))
}

