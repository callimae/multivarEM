library(parallel)
library(Matrix)

#' Generator mieszaniny rozkładów normalnych zoptymalizowany dla wysokich wymiarów
#' 
#' @param n Liczba obserwacji
#' @param cl Liczba klastrów
#' @param d Liczba wymiarów
#' @param separation Parametr kontrolujący separację średnich (0-1: nakładające się, >1: dobrze odseparowane)
#' @param seed Ziarno generatora liczb losowych
#' @param variance_pattern Wzorzec wariancji: "equal", "random", "decay", "sparse"
#' @param sparsity Rzadkość danych (tylko dla wysokich wymiarów)
#' @param n_cores Liczba rdzeni do obliczeń równoległych (0 = wszystkie dostępne)
#' @param store_params Czy przechowywać pełne parametry (false dla wysokich wymiarów)
#' @return Lista zawierająca parametry mieszaniny i wygenerowane dane
high_dim_mixture <- function(n = 1000, 
                             cl = 2, 
                             d = 1000, 
                             separation = 1.0,
                             seed = NULL,
                             variance_pattern = "equal",
                             sparsity = 0.0,
                             n_cores = 0,
                             store_params = FALSE) {
    
    # Ustawienie ziarna dla powtarzalności wyników
    if (!is.null(seed)) set.seed(seed)
    
    # Ustawienie liczby rdzeni do obliczeń równoległych
    if (n_cores == 0) {
        n_cores <- parallel::detectCores() - 1
        if (is.na(n_cores) || n_cores < 1) n_cores <- 1
    }
    
    # Generacja wag mieszaniny
    mixing.proportions <- runif(n = cl, min = 0.1)
    mixing.proportions <- mixing.proportions/sum(mixing.proportions)
    
    mixing.n <- round(mixing.proportions * n, 0)
    # Korekta dla błędów zaokrąglenia
    if (sum(mixing.n) != n) {
        diff <- n - sum(mixing.n)
        mixing.n[which.max(mixing.proportions)] <- mixing.n[which.max(mixing.proportions)] + diff
    }
    
    # W wysokich wymiarach wykorzystujemy strategię rzadkich różnic
    message("Generowanie średnich dla ", cl, " klastrów w ", d, " wymiarach...")
    
    # Klastry różnią się tylko w niewielkiej liczbie wymiarów dla efektywności obliczeniowej
    active_dims <- min(d, 100)  # Ograniczamy liczbę "aktywnych" wymiarów
    
    # Wybieramy losowo aktywne wymiary, jeśli d > 100
    if (d > 100) {
        active_dimensions <- sort(sample(1:d, active_dims))
    } else {
        active_dimensions <- 1:d
    }
    
    # Tworzymy bazowe wektory średnich - zoptymalizowane dla pamięci
    # Zamiast przechowywać pełne wektory, przechowujemy tylko niezerowe elementy
    means_info <- list()
    
    # Generowanie średnich
    if (d <= 100) {
        # Dla niższych wymiarów możemy używać poprzedniego podejścia
        mean_base <- rep(0, d)
        means <- list()
        
        for (k in 1:cl) {
            if (k == 1) {
                # Pierwszy klaster w początku układu
                means[[k]] <- mean_base
            } else {
                # Pozostałe klastry są przesunięte
                current_mean <- mean_base
                
                # Przesunięcie tylko w aktywnych wymiarach
                shift_dims <- sample(active_dimensions, min(k, length(active_dimensions)))
                current_mean[shift_dims] <- rnorm(length(shift_dims), 
                                                  mean = 0, 
                                                  sd = separation * 2)
                means[[k]] <- current_mean
            }
        }
        
        means_info <- means
    } else {
        # Dla bardzo wysokich wymiarów przechowujemy tylko różnice
        means_info <- vector("list", cl)
        means_info[[1]] <- numeric(0)  # Pierwszy klaster w początku układu, bez przesunięcia
        
        for (k in 2:cl) {
            # Dla każdego klastra przechowujemy tylko różnice w kilku wymiarach
            num_diff_dims <- min(30, d)  # Ustaw kilka różnic dla każdego klastra
            diff_dims <- sample(1:d, num_diff_dims)
            diff_values <- rnorm(num_diff_dims, mean = 0, sd = separation * 2)
            
            # Przechowujemy tylko indeksy i wartości przesunięć
            means_info[[k]] <- list(
                dims = diff_dims,
                values = diff_values
            )
        }
    }
    
    # Definiujemy funkcję do generowania danych dla jednego klastra
    generate_cluster_data <- function(k) {
        cluster_size <- mixing.n[k]
        
        if (d <= 100) {
            # Dla niższych wymiarów generujemy pełne dane
            if (variance_pattern == "equal") {
                # Równe wariancje we wszystkich wymiarach
                cluster_data <- matrix(rnorm(cluster_size * d), nrow = cluster_size)
                browser()
                # Dodaj średnią klastra
                cluster_data <- sweep(cluster_data, 2, means_info[[k]], "+")
            } else if (variance_pattern == "random") {
                # Losowe wariancje w różnych wymiarach
                vars <- runif(d, 0.5, 2.0)
                cluster_data <- matrix(rnorm(cluster_size * d), nrow = cluster_size)
                for (j in 1:d) {
                    cluster_data[, j] <- cluster_data[, j] * sqrt(vars[j])
                }
                
                # Dodaj średnią klastra
                cluster_data <- sweep(cluster_data, 2, means_info[[k]], "+")
            } else if (variance_pattern == "decay") {
                # Wariancje malejące wykładniczo (dla zmiennych skorelowanych)
                decay_rate <- 0.99
                vars <- decay_rate^(0:(d-1))
                cluster_data <- matrix(rnorm(cluster_size * d), nrow = cluster_size)
                for (j in 1:d) {
                    cluster_data[, j] <- cluster_data[, j] * sqrt(vars[j])
                }
                
                # Dodaj średnią klastra
                cluster_data <- sweep(cluster_data, 2, means_info[[k]], "+")
            } else if (variance_pattern == "sparse") {
                # Rzadkie dane (wiele zer)
                cluster_data <- matrix(0, nrow = cluster_size, ncol = d)
                zero_prob <- sparsity
                
                for (i in 1:cluster_size) {
                    non_zero_dims <- which(runif(d) > zero_prob)
                    cluster_data[i, non_zero_dims] <- rnorm(length(non_zero_dims))
                }
                
                # Dodaj średnią klastra
                cluster_data <- sweep(cluster_data, 2, means_info[[k]], "+")
            }
        } else {
            # Dla bardzo wysokich wymiarów używamy efektywniejszego podejścia
            # Bazowe dane z rozkładu normalnego
            cluster_data <- matrix(rnorm(cluster_size * d), nrow = cluster_size)
            
            if (variance_pattern == "decay") {
                # Zastosuj malejące wariancje dla pierwszych kilkuset wymiarów
                decay_dims <- min(d, 500)
                decay_rate <- 0.99
                vars <- decay_rate^(0:(decay_dims-1))
                for (j in 1:decay_dims) {
                    cluster_data[, j] <- cluster_data[, j] * sqrt(vars[j])
                }
            } else if (variance_pattern == "sparse") {
                # Utwórz rzadkie dane
                zero_mask <- matrix(runif(cluster_size * d) < sparsity, 
                                    nrow = cluster_size, ncol = d)
                cluster_data[zero_mask] <- 0
            }
            
            # Dodaj przesunięcie średniej tylko tam, gdzie jest to konieczne
            if (k > 1) {
                diff_dims <- means_info[[k]]$dims
                diff_values <- means_info[[k]]$values
                for (j in 1:length(diff_dims)) {
                    cluster_data[, diff_dims[j]] <- cluster_data[, diff_dims[j]] + diff_values[j]
                }
            }
        }
        
        return(cluster_data)
    }
    
    # Generujemy dane dla każdego klastra równolegle
    message("Generowanie danych dla klastrów...")
    if (n_cores > 1 && cl > 1) {
        # Równoległe generowanie danych
        cl_parallel <- parallel::makeCluster(min(n_cores, cl))
        parallel::clusterExport(cl_parallel, 
                                varlist = c("mixing.n", "means_info", "d", 
                                            "variance_pattern", "sparsity"),
                                envir = environment())
        
        clusters_data <- parallel::parLapply(cl_parallel, 1:cl, generate_cluster_data)
        parallel::stopCluster(cl_parallel)
    } else {
        # Sekwencyjne generowanie danych
        clusters_data <- lapply(1:cl, generate_cluster_data)
    }
    
    # Łączymy dane ze wszystkich klastrów
    message("Łączenie danych...")
    data_combined <- do.call(rbind, clusters_data)
    
    # Dodajemy informację o klastrach
    cluster_info <- rep(1:cl, times = mixing.n)
    
    # Przygotowanie wynikowej listy
    result <- list(
        "mixing_proportions" = mixing.proportions,
        "mixing.n" = mixing.n,
        "clusters" = cl,
        "dimensions" = d,
        "separation" = separation,
        "variance_pattern" = variance_pattern,
        "data" = data_combined,
        "cluster" = cluster_info
    )
    
    # Dodaj pełne parametry tylko gdy jest to żądane i wymiar nie jest zbyt duży
    if (store_params && d <= 100) {
        result$means <- means_info
    } else if (d > 100) {
        message("Wymiar danych zbyt duży, nie przechowuję pełnych parametrów.")
    }
    
    class(result) <- "high_dim_mixture"
    return(result)
}

#' Funkcja do projektowania wysokowymiarowych danych do wizualizacji
#' 
#' @param x Obiekt wysokowymiarowej mieszaniny
#' @param method Metoda projekcji: "pca", "random"
#' @param dims Liczba wymiarów do projekcji
#' @return Dane w niższym wymiarze
project_high_dim <- function(x, method = "pca", dims = 2) {
    if (!inherits(x, "high_dim_mixture")) {
        stop("Obiekt musi być klasy 'high_dim_mixture'")
    }
    
    data <- x$data
    
    if (method == "pca") {
        message("Wykonywanie PCA...")
        # Możemy użyć prcomp, ale dla dużych zestawów danych może być zbyt kosztowne
        # Obliczmy pierwszych kilka składowych głównych
        
        if (ncol(data) > 10000 || nrow(data) > 10000) {
            message("Duży zestaw danych, używam wydajniejszej implementacji PCA...")
            # Używamy irlba dla dużych zestawów danych - przybliżone SVD
            if (!requireNamespace("irlba", quietly = TRUE)) {
                stop("Pakiet 'irlba' jest wymagany dla efektywnej redukcji wymiarowości. Zainstaluj go używając: install.packages('irlba')")
            }
            
            # Oblicz średnią kolumn
            col_means <- colMeans(data)
            centered_data <- sweep(data, 2, col_means, "-")
            
            # Oblicz przybliżone SVD
            svd_result <- irlba::irlba(centered_data, nv = dims)
            
            # Projektuj dane
            projected_data <- centered_data %*% svd_result$v
        } else {
            # Standardowe PCA dla mniejszych zestawów danych
            pca_result <- prcomp(data, center = TRUE, scale. = FALSE)
            projected_data <- pca_result$x[, 1:min(dims, ncol(pca_result$x)), drop = FALSE]
        }
    } else if (method == "random") {
        message("Wykonywanie losowej projekcji...")
        # Prosta losowa projekcja
        d <- ncol(data)
        projection_matrix <- matrix(rnorm(d * dims), nrow = d) / sqrt(dims)
        projected_data <- data %*% projection_matrix
    } else {
        stop("Nieznana metoda projekcji. Użyj 'pca' lub 'random'.")
    }
    
    # Przygotuj wynikowy data frame
    result <- as.data.frame(projected_data)
    names(result) <- paste0("PC", 1:ncol(projected_data))
    result$cluster <- factor(x$cluster)
    
    return(result)
}

#' Funkcja do wizualizacji wysokowymiarowych mieszanin
#' 
#' @param x Obiekt wysokowymiarowej mieszaniny
#' @param method Metoda projekcji: "pca", "random"
#' @param dims Wymiary do wizualizacji (np. c(1,2) dla PC1 i PC2)
#' @return Wykres ggplot2
plot.high_dim_mixture <- function(x, method = "pca", dims = c(1, 2), ...) {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("Pakiet ggplot2 jest wymagany dla wizualizacji")
    }
    
    # Projektuj dane
    projected <- project_high_dim(x, method = method, dims = max(dims))
    
    # Wybierz wymiary do wizualizacji
    x_col <- paste0("PC", dims[1])
    y_col <- paste0("PC", dims[2])
    
    # Stwórz wykres
    p <- ggplot2::ggplot(projected, ggplot2::aes_string(x = x_col, y = y_col, color = "cluster")) +
        ggplot2::geom_point(alpha = 0.7) +
        ggplot2::labs(title = paste("Mieszanina", x$clusters, "rozkładów normalnych w", x$dimensions, "wymiarach"),
                      subtitle = paste("Separacja:", x$separation, "| Metoda projekcji:", method),
                      x = paste("Wymiar", dims[1]),
                      y = paste("Wymiar", dims[2])) +
        ggplot2::theme_minimal()
    
    return(p)
}

# Przykład użycia:
system.time({
  dane_wysokie <- high_dim_mixture(n = 2000, cl = 4, d = 1000,
                                  separation = 0.4,
                                  variance_pattern = "sparse",
                                  n_cores = 1)
})

plot(dane_wysokie, method = "pca")
d <- dist(x = dane_wysokie$data, method = "manhattan")
hc <- hclust(d, method = "ward.D")
mclust::adjustedRandIndex(cutree(hc, 4), dane_wysokie$cluster)
mc <- mclust::Mclust(dane_wysokie$data, G = 4)
mclust::adjustedRandIndex(mc$classification, dane_wysokie$cluster)
gmm <- GaussEM(dane_wysokie$data, k = 4, ini = "kmeans++", em.itr = 3000, kmeans_init = 20)
mclust::adjustedRandIndex(gmm$cluster, dane_wysokie$cluster)
