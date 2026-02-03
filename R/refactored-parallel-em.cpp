// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// Wspólne funkcje narzędziowe (bez przestrzeni nazw)
// ---------------------------------------------------------------------

// Binary search function for k-means++ initialization
std::size_t binary_search_cumsum(const arma::vec& cumsum, double value) {
  std::size_t left = 0;
  std::size_t right = cumsum.n_elem - 1;
  
  while (left < right) {
    std::size_t mid = left + (right - left) / 2;
    if (cumsum(mid) < value) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  
  return left;
}

// Function for safe row normalization of matrix with handling of zero sums
arma::mat safe_row_normalize(const arma::mat& input_matrix, double epsilon = 1e-10) {
  std::size_t rows = input_matrix.n_rows;
  std::size_t cols = input_matrix.n_cols;
  arma::mat normalized(rows, cols);

  for (std::size_t i = 0; i < rows; i++) {
    double row_sum = sum(input_matrix.row(i));

    if (row_sum > epsilon) {
      // Normalize by row sum when sum is large enough
      normalized.row(i) = input_matrix.row(i) / row_sum;
    } else {
      // When sum is close to zero, use uniform distribution with small noise
      for (std::size_t j = 0; j < cols; j++) {
        normalized(i, j) = (1.0 / cols) + R::runif(-1e-5, 1e-5) * (1.0 / cols);
      }
      // Re-normalize to ensure sum = 1
      normalized.row(i) = normalized.row(i) / sum(normalized.row(i));
    }

    // Handle zero values - replace with very small numbers
    for (std::size_t j = 0; j < cols; j++) {
      if (normalized(i, j) < epsilon) {
        normalized(i, j) = epsilon;
      }
    }

    // Re-normalize after adding epsilon
    normalized.row(i) = normalized.row(i) / sum(normalized.row(i));
  }

  return normalized;
}

// Namespace for common clustering functionality (wspólne dla obu algorytmów)
// ---------------------------------------------------------------------
namespace ClusteringCommon {

// Structure for parallel distance calculation
struct DistanceWorker : public Worker {
  // Input data
  const arma::mat& data;
  const arma::mat& centers;
  
  // Result
  arma::mat& distances;
  
  // Constructor
  DistanceWorker(const arma::mat& data, const arma::mat& centers, arma::mat& distances)
    : data(data), centers(centers), distances(distances) {}
  
  // Computational operator - Optimized to use squared distances
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for (std::size_t j = 0; j < centers.n_rows; j++) {
        // Use accu(square()) instead of norm() to avoid square root calculation
        distances(i, j) = arma::accu(arma::square(data.row(i) - centers.row(j)));
      }
    }
  }
};

// Structure for parallel cluster assignment
struct AssignClusterWorker : public Worker {
  // Input data
  const arma::mat& distances;
  
  // Result
  arma::uvec& clusters;
  
  // Constructor
  AssignClusterWorker(const arma::mat& distances, arma::uvec& clusters)
    : distances(distances), clusters(clusters) {}
  
  // Computational operator
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      clusters(i) = distances.row(i).index_min();
    }
  }
};

// Structure for direct cluster assignment (without precomputed distances)
struct DirectAssignClusterWorker : public Worker {
  // Input data
  const arma::mat& data;
  const arma::mat& centers;

  // Result
  arma::uvec& clusters;

  // Constructor
  DirectAssignClusterWorker(const arma::mat& data, const arma::mat& centers, arma::uvec& clusters)
    : data(data), centers(centers), clusters(clusters) {}

  // Computational operator - Optimized for squared distances
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      double min_dist = std::numeric_limits<double>::max();
      std::size_t best_cluster = 0;

      // Find the closest center for point i
      for (std::size_t j = 0; j < centers.n_rows; j++) {
        // Use squared distance directly
        double dist = arma::accu(arma::square(data.row(i) - centers.row(j)));
        if (dist < min_dist) {
          min_dist = dist;
          best_cluster = j;
        }
      }

      clusters(i) = best_cluster;
    }
  }
};

// Structure for parallel minimum distance calculation
struct MinDistanceWorker : public Worker {
  // Input data
  const arma::mat& data;
  const arma::mat& centers;
  const std::size_t centers_count;
  
  // Result
  arma::vec& min_distances;
  
  // Constructor
  MinDistanceWorker(const arma::mat& data, const arma::mat& centers, std::size_t centers_count, arma::vec& min_distances)
    : data(data), centers(centers), centers_count(centers_count), min_distances(min_distances) {}
  
  // Computational operator - Optimized to use squared distances directly
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      double min_dist = std::numeric_limits<double>::infinity();
      
      for (std::size_t c = 0; c < centers_count; c++) {
        // Use squared distance directly
        double dist = arma::accu(arma::square(data.row(i) - centers.row(c)));
        min_dist = std::min(min_dist, dist);
      }
      
      min_distances(i) = min_dist; // Already squared distance
    }
  }
};

// Structure for parallel WCSS calculation
struct WCSSWorker : public Worker {
  // Input data
  const arma::mat& data;
  const arma::uvec& clusters;
  const arma::mat& centers;
  
  // Result
  double& wcss;
  std::mutex& mutex;
  
  // Constructor
  WCSSWorker(const arma::mat& data, const arma::uvec& clusters, const arma::mat& centers, double& wcss, std::mutex& mutex)
    : data(data), clusters(clusters), centers(centers), wcss(wcss), mutex(mutex) {}
  
  // Computational operator - Optimized to use squared distances directly
  void operator()(std::size_t begin, std::size_t end) {
    double local_wcss = 0.0;
    
    for (std::size_t i = begin; i < end; i++) {
      // Use squared distance directly
      local_wcss += arma::accu(arma::square(data.row(i) - centers.row(clusters(i))));
    }
    
    // Mutex lock for shared value update
    std::lock_guard<std::mutex> lock(mutex);
    wcss += local_wcss;
  }
};

// Structure for parallel cluster means calculation
struct ClusterMeansWorker : public Worker {
  // Input data
  const arma::mat& data;
  const arma::uvec& clusters;
  const arma::uvec& unique_clusters;
  
  // Result
  arma::mat& means;
  
  // Constructor
  ClusterMeansWorker(const arma::mat& data, const arma::uvec& clusters,
                     const arma::uvec& unique_clusters, arma::mat& means)
    : data(data), clusters(clusters), unique_clusters(unique_clusters), means(means) {}
  
  // Computational operator
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      unsigned int cluster_id = unique_clusters(i);
      arma::uvec idx = find(clusters == cluster_id);
      
      if (idx.n_elem > 0) {
        means.row(i) = mean(data.rows(idx), 0);
      }
    }
  }
};

// Function for parallel distance calculation
void compute_distances_parallel(const arma::mat& data, const arma::mat& centers, arma::mat& distances) {
  DistanceWorker worker(data, centers, distances);
  parallelFor(0, data.n_rows, worker);
}

// Function for parallel cluster assignment
void assign_clusters_parallel(const arma::mat& distances, arma::uvec& clusters) {
  AssignClusterWorker worker(distances, clusters);
  parallelFor(0, distances.n_rows, worker);
}

// Function for direct parallel cluster assignment (without precomputed distances matrix)
void direct_assign_clusters_parallel(const arma::mat& data, const arma::mat& centers, arma::uvec& clusters) {
  DirectAssignClusterWorker worker(data, centers, clusters);
  parallelFor(0, data.n_rows, worker);
}

// Function for parallel minimum distance calculation
void compute_min_distances_parallel(const arma::mat& data, const arma::mat& centers, std::size_t centers_count, arma::vec& min_distances) {
  MinDistanceWorker worker(data, centers, centers_count, min_distances);
  parallelFor(0, data.n_rows, worker);
}

// Helper function for parallel cluster means calculation
arma::mat parallel_cluster_means(const arma::mat& data, const arma::uvec& clusters,
                                const arma::uvec& unique_clusters) {
  arma::mat means(unique_clusters.n_elem, data.n_cols, fill::zeros);
  
  // Use parallel processing only for large datasets or many clusters
  if (unique_clusters.n_elem > 4 || data.n_rows > 5000) {
    ClusterMeansWorker worker(data, clusters, unique_clusters, means);
    parallelFor(0, unique_clusters.n_elem, worker);
  } else {
    // For a small number of clusters, execute sequentially
    for (std::size_t i = 0; i < unique_clusters.n_elem; i++) {
      unsigned int cluster_id = unique_clusters(i);
      arma::uvec idx = find(clusters == cluster_id);
      
      if (idx.n_elem > 0) {
        means.row(i) = mean(data.rows(idx), 0);
      }
    }
  }
  
  return means;
}

// Function for random cluster initialization with guaranteed coverage
arma::uvec random_clusters(std::size_t n, int k) {
  arma::uvec clusters = randi<uvec>(n, distr_param(0, k-1));
  
  // Ensure each cluster has at least one observation
  arma::uvec unique_vals = unique(clusters);
  int attempts = 0;
  const int max_attempts = 100; // Prevent infinite loop
  
  while (unique_vals.n_elem < static_cast<size_t>(k) && attempts < max_attempts) {
    // If some clusters are missing, try again
    clusters = randi<uvec>(n, distr_param(0, k-1));
    unique_vals = unique(clusters);
    attempts++;
  }
  
  // Force allocation if still not all clusters represented
  if (unique_vals.n_elem < static_cast<size_t>(k)) {
    for (int i = 0; i < k; i++) {
      bool cluster_exists = false;
      for (std::size_t j = 0; j < unique_vals.n_elem; j++) {
        if (unique_vals(j) == static_cast<arma::uword>(i)) {
          cluster_exists = true;
          break;
        }
      }
      
      if (!cluster_exists) {
        // Assign at least one point to this cluster
        clusters(i % n) = i;
      }
    }
  }
  
  return clusters;
}

// Common implementation of K-means/K-means++
// To avoid redundancy between GaussEM and MultiEM implementations
List kmeans_common(const arma::mat& data, int k, bool use_kmeans_plus_plus = true, 
                  int max_iter = 25, int n_init = 3) {
  std::size_t n = data.n_rows;
  std::size_t d = data.n_cols;
  
  double best_wcss = std::numeric_limits<double>::infinity();
  arma::mat best_centers;
  arma::uvec best_clusters;
  
  for (int init = 0; init < n_init; init++) {
    // Initialize centers
    arma::mat centers(k, d);
    
    if (use_kmeans_plus_plus) {
      // K-means++ initialization
      // Select first center randomly
      std::size_t first_idx = arma::randi<arma::uword>(arma::distr_param(0, n-1));
      centers.row(0) = data.row(first_idx);
      
      // Select remaining centers using k-means++
      for (int j = 1; j < k; j++) {
        // Calculate distance of each point to the nearest center
        arma::vec min_distances(n);
        compute_min_distances_parallel(data, centers, j, min_distances);
        
        // Check if sum of distances is close to zero
        double sum_distances = arma::sum(min_distances);
        if (sum_distances < 1e-10) {
          // If all points are close to centers, choose randomly
          std::size_t rand_idx = arma::randi<arma::uword>(arma::distr_param(0, n-1));
          centers.row(j) = data.row(rand_idx);
          continue;
        }
        
        // Sample new center with probability proportional to distance
        arma::vec probs = min_distances / sum_distances;
        
        // Use cumulative sum for efficient sampling
        arma::vec cumsum_probs = arma::cumsum(probs);
        
        // Random value for sampling
        double rand_val = R::runif(0, 1);
        
        // Binary search to find the next center
        std::size_t next_center = binary_search_cumsum(cumsum_probs, rand_val);
        
        centers.row(j) = data.row(next_center);
      }
    } else {
      // Regular k-means with random initialization
      arma::uvec indices = arma::randperm(n, k);
      centers = data.rows(indices);
    }
    
    // Run standard k-means with these centers
    arma::uvec clusters(n);
    bool converged = false;
    int iter = 0;
    
    while (!converged && iter < max_iter) {
      // Assign points to nearest centers directly (without storing full distance matrix)
      arma::uvec new_clusters(n);
      direct_assign_clusters_parallel(data, centers, new_clusters);
      
      // Check convergence
      if (arma::all(new_clusters == clusters)) {
        converged = true;
      } else {
        clusters = new_clusters;
      }
      
      // Update centers
      arma::mat new_centers(k, d, fill::zeros);
      arma::uvec counts(k, fill::zeros);
      
      for (std::size_t i = 0; i < n; i++) {
        new_centers.row(clusters(i)) += data.row(i);
        counts(clusters(i))++;
      }
      
      // Handle empty clusters
      for (int j = 0; j < k; j++) {
        if (counts(j) > 0) {
          new_centers.row(j) /= counts(j);
        } else {
          // If cluster is empty, use the point furthest from its center
          arma::vec distances(n);
          
          // Calculate distances to current centers
          for (std::size_t i = 0; i < n; i++) {
            distances(i) = arma::accu(arma::square(data.row(i) - centers.row(clusters(i))));
          }
          
          std::size_t furthest_idx = distances.index_max();
          new_centers.row(j) = data.row(furthest_idx);
        }
      }
      
      centers = new_centers;
      iter++;
    }
    
    // Calculate WCSS in parallel
    double wcss = 0.0;
    std::mutex mutex;
    WCSSWorker worker(data, clusters, centers, wcss, mutex);
    parallelFor(0, n, worker);
    
    if (wcss < best_wcss) {
      best_wcss = wcss;
      best_centers = centers;
      best_clusters = clusters;
    }
  }
  
  return List::create(
    Named("centers") = best_centers,
    Named("cluster") = best_clusters + 1, // +1 for R indexing
    Named("wcss") = best_wcss
  );
}

} // End namespace ClusteringCommon

// Namespace for Gaussian EM functionality
// ---------------------------------------------------------------------
namespace GaussEM {

// Korzystanie ze wspólnych funkcji z ClusteringCommon
using ClusteringCommon::parallel_cluster_means;
using ClusteringCommon::compute_distances_parallel;
using ClusteringCommon::assign_clusters_parallel;
using ClusteringCommon::compute_min_distances_parallel;
using ClusteringCommon::random_clusters;
using ClusteringCommon::kmeans_common;

// Parallel K-means implementation (wrapper around common implementation)
List parallel_kmeans(const arma::mat& data, int k, int max_iter = 25, int n_init = 3) {
  return ClusteringCommon::kmeans_common(data, k, false, max_iter, n_init);
}

// Parallel K-means++ implementation (wrapper around common implementation)
List parallel_kmeans_plusplus(const arma::mat& data, int k, int max_iter = 25, int n_init = 3) {
  return ClusteringCommon::kmeans_common(data, k, true, max_iter, n_init);
}

} // End namespace GaussEM

// Namespace for Multinomial EM functionality
// ---------------------------------------------------------------------
namespace MultiEM {

// Korzystanie ze wspólnych funkcji z ClusteringCommon
using ClusteringCommon::compute_distances_parallel;
using ClusteringCommon::assign_clusters_parallel;
using ClusteringCommon::compute_min_distances_parallel;
using ClusteringCommon::direct_assign_clusters_parallel;
using ClusteringCommon::kmeans_common;

// Structure for parallel log-posterior calculation
struct LogPosteriorWorker : public Worker {
  // Input data
  const arma::mat& xdata;
  const arma::mat& safe_probs;
  const arma::vec& alphas;
  double epsilon;

  // Result
  arma::mat& log_post;

  // Constructor
  LogPosteriorWorker(const arma::mat& xdata, const arma::mat& safe_probs, 
                    const arma::vec& alphas, double epsilon, arma::mat& log_post)
    : xdata(xdata), safe_probs(safe_probs), alphas(alphas), epsilon(epsilon), log_post(log_post) {}

  // Computational operator
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for (std::size_t j = 0; j < safe_probs.n_rows; j++) {
        double log_prob = 0.0;

        for (std::size_t c = 0; c < xdata.n_cols; c++) {
          double x = xdata(i, c);
          double p = safe_probs(j, c);

          if (x > 0 && p > 0) {
            log_prob += x * std::log(p);
          }
        }

        double alpha = std::max(alphas(j), epsilon);
        log_post(i, j) = log_prob + std::log(alpha);
      }
    }
  }
};

// Optimized K-means++ implementation (wrapper around common implementation)
List optimized_kmeans_plusplus(const arma::mat& data, int k, int max_iter = 25, int n_init = 3) {
  return ClusteringCommon::kmeans_common(data, k, true, max_iter, n_init);
}

// Parallel implementation of log-posterior calculation
arma::mat parallel_compute_log_posteriors(const arma::mat& xdata,
                                         const arma::mat& probs,
                                         const arma::vec& alphas,
                                         double epsilon = 1e-10) {
  std::size_t n = xdata.n_rows;
  std::size_t k = probs.n_rows;
  arma::mat log_post(n, k);

  // Safe guard against zeros in probs
  arma::mat safe_probs = probs;
  for (std::size_t i = 0; i < probs.n_rows; i++) {
    for (std::size_t j = 0; j < probs.n_cols; j++) {
      if (probs(i, j) < epsilon) {
        safe_probs(i, j) = epsilon;
      }
    }
  }

  // Use parallel processing for large datasets
  if (n > 1000) {
    LogPosteriorWorker worker(xdata, safe_probs, alphas, epsilon, log_post);
    parallelFor(0, n, worker);
  } else {
    // For smaller datasets, use sequential processing
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < k; j++) {
        double log_prob = 0.0;

        for (std::size_t c = 0; c < xdata.n_cols; c++) {
          double x = xdata(i, c);
          double p = safe_probs(j, c);

          if (x > 0 && p > 0) {
            log_prob += x * std::log(p);
          }
        }

        double alpha = std::max(alphas(j), epsilon);
        log_post(i, j) = log_prob + std::log(alpha);
      }
    }
  }

  return log_post;
}

} // End namespace MultiEM

// Exported R functions (outside of namespaces to maintain the same interface)
// ---------------------------------------------------------------------

// Parallel K-means implementation
// [[Rcpp::export]]
List parallel_kmeans(const arma::mat& data, int k, int max_iter = 25, int n_init = 3) {
  return GaussEM::parallel_kmeans(data, k, max_iter, n_init);
}

// Parallel K-means++ implementation
// [[Rcpp::export]]
List parallel_kmeans_plusplus(const arma::mat& data, int k, int max_iter = 25, int n_init = 3) {
  return GaussEM::parallel_kmeans_plusplus(data, k, max_iter, n_init);
}

// Main GMM initialization function
// [[Rcpp::export]]
List gaussEM_ini_cpp_parallel(const arma::mat& xdata, int k,
                              std::string ini = "random",
                              int kmeans_iter = 25,
                              int kmeans_init = 3,
                              double epsilon = 1e-6) {
  std::size_t n = xdata.n_rows;
  std::size_t d = xdata.n_cols;
  
  arma::mat means(k, d);
  arma::mat vars(k, d);
  arma::vec alphas(k);
  
  // Check input data for missing values and outliers
  bool has_na = false;
  bool has_inf = false;
  
  // Vectorized check for NAs and Infs
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      if (std::isnan(xdata(i, j))) {
        has_na = true;
      }
      if (!std::isfinite(xdata(i, j))) {
        has_inf = true;
      }
    }
  }
  
  if (has_na) {
    warning("Data contains missing values (NA). This may affect the quality of results.");
  }
  
  if (has_inf) {
    warning("Data contains infinite values (Inf). This may affect the quality of results.");
  }
  
  if (ini == "random") {
    // Random initialization
    arma::uvec clusters = ClusteringCommon::random_clusters(n, k);
    arma::uvec unique_clusters = unique(clusters);
    
    // Calculate means for each cluster in parallel
    means = ClusteringCommon::parallel_cluster_means(xdata, clusters, unique_clusters);
    
    // Random initialization of weights in range [0.05, 1.0]
    alphas = arma::randu<arma::vec>(k) * 0.95 + 0.05;
    
    // Normalize weights
    alphas = alphas / sum(alphas);
    
  } else if (ini == "kmeans") {
    // Use parallel K-means
    List km_result = GaussEM::parallel_kmeans(xdata, k, kmeans_iter, kmeans_init);
    
    arma::mat centers = as<arma::mat>(km_result["centers"]);
    arma::uvec clusters = as<arma::uvec>(km_result["cluster"]) - 1; // R uses 1-based indexing
    
    means = centers;
    
    // Calculate weights based on cluster sizes
    arma::vec cluster_counts(k, fill::zeros);
    for (std::size_t i = 0; i < n; i++) {
      cluster_counts(clusters(i))++;
    }
    alphas = cluster_counts / static_cast<double>(n);
    
    // Ensure minimum weight for each cluster
    for (int i = 0; i < k; i++) {
      if (alphas(i) < epsilon) {
        alphas(i) = epsilon;
      }
    }
    
    // Renormalize weights
    alphas = alphas / sum(alphas);
    
  } else if (ini == "kmeans++") {
    // Use parallel K-means++
    List km_result = GaussEM::parallel_kmeans_plusplus(xdata, k, kmeans_iter, kmeans_init);
    
    arma::mat centers = as<arma::mat>(km_result["centers"]);
    arma::uvec clusters = as<arma::uvec>(km_result["cluster"]) - 1; // R uses 1-based indexing
    
    means = centers;
    
    // Calculate weights based on cluster sizes
    arma::vec cluster_counts(k, fill::zeros);
    for (std::size_t i = 0; i < n; i++) {
      cluster_counts(clusters(i))++;
    }
    alphas = cluster_counts / static_cast<double>(n);
    
    // Ensure minimum weight for each cluster
    for (int i = 0; i < k; i++) {
      if (alphas(i) < epsilon) {
        alphas(i) = epsilon;
      }
    }
    
    // Renormalize weights
    alphas = alphas / sum(alphas);
    
  } else {
    stop("Unknown initialization method: " + ini);
  }
  
  // Initialize variances
  vars = arma::mat(k, d);
  
  // Avoid zeros in variances
  for (int i = 0; i < k; i++) {
    for (std::size_t j = 0; j < d; j++) {
      // Use absolute value of mean + constant
      vars(i, j) = std::max(std::abs(means(i, j)) + 0.1, epsilon);
      
      // Ensure variance is not zero or NaN
      if (vars(i, j) <= 0 || std::isnan(vars(i, j))) {
        // Calculate global variance of this feature as fallback
        double global_var = arma::var(xdata.col(j));
        
        // If global variance is also problematic, use constant
        if (global_var <= 0 || std::isnan(global_var)) {
          vars(i, j) = 1.0;
        } else {
          vars(i, j) = global_var;
        }
      }
    }
  }
  
  // Handle NaN and invalid values - in parallel
  for (int i = 0; i < k; i++) {
    for (std::size_t j = 0; j < d; j++) {
      if (std::isnan(vars(i, j)) || vars(i, j) <= epsilon) {
        // Calculate global variance of this feature
        double global_var = arma::var(xdata.col(j));
        
        // If global variance is also problematic, use random value
        if (std::isnan(global_var) || global_var <= epsilon) {
          vars(i, j) = R::runif(epsilon, 1.0);
        } else {
          vars(i, j) = global_var;
        }
      }
    }
  }
  
  arma::mat varmin = abs(vars);
  
  // Additional check for parameter validity
  for (int i = 0; i < k; i++) {
    for (std::size_t j = 0; j < d; j++) {
      if (std::isnan(means(i, j))) {
        means(i, j) = arma::mean(xdata.col(j));
      }
      if (std::isnan(varmin(i, j)) || varmin(i, j) <= 0) {
        varmin(i, j) = epsilon;
      }
    }
  }
  
  // Return results in exactly the same format as the original function
  return List::create(
    Named("means") = means,
    Named("vars") = vars,
    Named("varmin") = varmin,
    Named("alphas") = alphas
  );
}

// Optimized K-means++ implementation using RcppParallel
// [[Rcpp::export]]
List optimized_kmeans_plusplus(const arma::mat& data, int k, int max_iter = 25, int n_init = 3) {
  return MultiEM::optimized_kmeans_plusplus(data, k, max_iter, n_init);
}

// Optimized MultiEM initialization function with handling of zero probabilities
// [[Rcpp::export]]
List improved_multiEM_ini(const arma::mat& xdata, int k,
                          std::string ini_type = "random",
                          int kmeans_iter = 25,
                          int kmeans_init = 3,
                          double smoothing = 1e-10) {
  std::size_t n = xdata.n_rows;
  std::size_t d = xdata.n_cols;

  arma::mat prob_mat;

  if (ini_type == "random") {
    // Random initialization (using vectorization)
    prob_mat = arma::mat(k, d);

    // Inicjalizacja losowa
    for (int i = 0; i < k; i++) {
      for (std::size_t j = 0; j < d; j++) {
        // Add small value to each element to avoid zeros
        prob_mat(i, j) = R::rpois(d) + smoothing;
      }
    }

  } else if (ini_type == "kmeans" || ini_type == "kmeans++") {
    // Use optimized K-means++
    List km_result = MultiEM::optimized_kmeans_plusplus(xdata, k, kmeans_iter, kmeans_init);
    arma::mat centers = as<arma::mat>(km_result["centers"]);

    // Take absolute values from centers and add small value
    prob_mat = arma::abs(centers) + smoothing;

  } else {
    stop("Unknown initialization method: " + ini_type);
  }

  // Safe row normalization with handling of zero sums
  arma::mat prob_mat_st = safe_row_normalize(prob_mat, smoothing);

  // Initialize alphas (weight parameters)
  arma::vec alphas(k);

  for (int i = 0; i < k; i++) {
    // Initialize alphas avoiding values close to zero
    alphas(i) = R::runif(0.2, 1.0);
  }

  // Normalize weights
  alphas = alphas / arma::sum(alphas);

  return List::create(
    Named("probs") = prob_mat_st,
    Named("alphas") = alphas
  );
}

// Safe function for calculating log-posteriors with handling of zero probabilities
// [[Rcpp::export]]
arma::mat safe_compute_log_posteriors(const arma::mat& xdata,
                                      const arma::mat& probs,
                                      const arma::vec& alphas,
                                      double epsilon = 1e-10) {
  return MultiEM::parallel_compute_log_posteriors(xdata, probs, alphas, epsilon);
}

// Parallel implementation of log-posterior calculation
// [[Rcpp::export]]
arma::mat parallel_compute_log_posteriors(const arma::mat& xdata,
                                         const arma::mat& probs,
                                         const arma::vec& alphas,
                                         double epsilon = 1e-10) {
  return MultiEM::parallel_compute_log_posteriors(xdata, probs, alphas, epsilon);
}

// Namespace for Poisson EM functionality
// ---------------------------------------------------------------------
namespace PoissonEM {

// Use common functions from ClusteringCommon
using ClusteringCommon::compute_distances_parallel;
using ClusteringCommon::assign_clusters_parallel;
using ClusteringCommon::compute_min_distances_parallel;
using ClusteringCommon::direct_assign_clusters_parallel;
using ClusteringCommon::kmeans_common;

// Structure for parallel log-posterior calculation for Poisson distribution
struct PoissonLogPosteriorWorker : public Worker {
  // Input data
  const arma::mat& xdata;
  const arma::mat& lambdas;
  const arma::vec& alphas;
  double epsilon;

  // Result
  arma::mat& log_post;

  // Constructor
  PoissonLogPosteriorWorker(const arma::mat& xdata, const arma::mat& lambdas,
                           const arma::vec& alphas, double epsilon, arma::mat& log_post)
    : xdata(xdata), lambdas(lambdas), alphas(alphas), epsilon(epsilon), log_post(log_post) {}

  // Computational operator
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for (std::size_t j = 0; j < lambdas.n_rows; j++) {
        double log_prob = 0.0;

        for (std::size_t c = 0; c < xdata.n_cols; c++) {
          double x = xdata(i, c);
          double lambda = std::max(lambdas(j, c), epsilon); // Ensure lambda is positive

          // Poisson log-probability: x * log(lambda) - lambda - log(x!)
          // Note: log(x!) is constant for fixed x, can be omitted for posterior comparison
          if (x >= 0) {
            log_prob += x * std::log(lambda) - lambda;
            // Subtract log(x!) only if needed for exact probability
            // This term is constant for each x and cancels out in posterior calculation
            // If needed: log_prob -= std::lgamma(x + 1);
          } else {
            // Handle negative values (not valid for Poisson)
            log_prob += -1000.0; // Very low probability
          }
        }

        double alpha = std::max(alphas(j), epsilon);
        log_post(i, j) = log_prob + std::log(alpha);
      }
    }
  }
};

// Parallel implementation of log-posterior calculation for Poisson
arma::mat parallel_compute_poisson_log_posteriors(const arma::mat& xdata,
                                                const arma::mat& lambdas,
                                                const arma::vec& alphas,
                                                double epsilon = 1e-10) {
  std::size_t n = xdata.n_rows;
  std::size_t k = lambdas.n_rows;
  arma::mat log_post(n, k);

  // Ensure lambdas are positive
  arma::mat safe_lambdas = lambdas;
  for (std::size_t i = 0; i < lambdas.n_rows; i++) {
    for (std::size_t j = 0; j < lambdas.n_cols; j++) {
      if (lambdas(i, j) < epsilon) {
        safe_lambdas(i, j) = epsilon;
      }
    }
  }

  // Use parallel processing for large datasets
  if (n > 1000) {
    PoissonLogPosteriorWorker worker(xdata, safe_lambdas, alphas, epsilon, log_post);
    parallelFor(0, n, worker);
  } else {
    // For smaller datasets, use sequential processing
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < k; j++) {
        double log_prob = 0.0;

        for (std::size_t c = 0; c < xdata.n_cols; c++) {
          double x = xdata(i, c);
          double lambda = std::max(safe_lambdas(j, c), epsilon);

          if (x >= 0) {
            log_prob += x * std::log(lambda) - lambda;
          } else {
            log_prob += -1000.0;
          }
        }

        double alpha = std::max(alphas(j), epsilon);
        log_post(i, j) = log_prob + std::log(alpha);
      }
    }
  }

  return log_post;
}

} // End namespace PoissonEM

// Poisson EM initialization function
// [[Rcpp::export]]
List poissonEM_ini_cpp_parallel(const arma::mat& xdata, int k,
                              std::string ini = "random",
                              int kmeans_iter = 25,
                              int kmeans_init = 3,
                              double epsilon = 1e-6) {
  std::size_t n = xdata.n_rows;
  std::size_t d = xdata.n_cols;

  arma::mat lambdas(k, d);
  arma::vec alphas(k);

  // Check input data for missing values and negative values (Poisson requires non-negative values)
  bool has_na = false;
  bool has_negative = false;

  // Vectorized check for NAs and negative values
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      if (std::isnan(xdata(i, j))) {
        has_na = true;
      }
      if (xdata(i, j) < 0) {
        has_negative = true;
      }
    }
  }

  if (has_na) {
    warning("Data contains missing values (NA). This may affect the quality of results.");
  }

  if (has_negative) {
    warning("Data contains negative values. Poisson distribution requires non-negative integers.");
  }

  if (ini == "random") {
    // Random initialization
    arma::uvec clusters = ClusteringCommon::random_clusters(n, k);
    arma::uvec unique_clusters = unique(clusters);

    // Calculate means for each cluster in parallel - use as lambda values
    lambdas = ClusteringCommon::parallel_cluster_means(xdata, clusters, unique_clusters);

    // Ensure lambdas are positive
    for (int i = 0; i < k; i++) {
      for (std::size_t j = 0; j < d; j++) {
        if (lambdas(i, j) <= 0 || std::isnan(lambdas(i, j))) {
          // Use small positive value if lambda is non-positive or NaN
          lambdas(i, j) = epsilon + R::rexp(1.0);
        }
      }
    }

    // Random initialization of weights in range [0.05, 1.0]
    alphas = arma::randu<arma::vec>(k) * 0.95 + 0.05;

    // Normalize weights
    alphas = alphas / sum(alphas);

  } else if (ini == "kmeans" || ini == "kmeans++") {
    // Use parallel K-means++
    List km_result = ClusteringCommon::kmeans_common(xdata, k, true, kmeans_iter, kmeans_init);

    arma::mat centers = as<arma::mat>(km_result["centers"]);
    arma::uvec clusters = as<arma::uvec>(km_result["cluster"]) - 1; // R uses 1-based indexing

    lambdas = centers;

    // Ensure lambdas are positive
    for (int i = 0; i < k; i++) {
      for (std::size_t j = 0; j < d; j++) {
        if (lambdas(i, j) <= 0 || std::isnan(lambdas(i, j))) {
          // Use small positive value if lambda is non-positive or NaN
          lambdas(i, j) = epsilon + R::rexp(1.0);
        }
      }
    }

    // Calculate weights based on cluster sizes
    arma::vec cluster_counts(k, fill::zeros);
    for (std::size_t i = 0; i < n; i++) {
      cluster_counts(clusters(i))++;
    }
    alphas = cluster_counts / static_cast<double>(n);

    // Ensure minimum weight for each cluster
    for (int i = 0; i < k; i++) {
      if (alphas(i) < epsilon) {
        alphas(i) = epsilon;
      }
    }

    // Renormalize weights
    alphas = alphas / sum(alphas);

  } else {
    stop("Unknown initialization method: " + ini);
  }

  // Return results
  return List::create(
    Named("lambdas") = lambdas,
    Named("alphas") = alphas
  );
}

// Function for parallel log-posterior calculation for Poisson distribution
// [[Rcpp::export]]
arma::mat parallel_compute_poisson_log_posteriors(const arma::mat& xdata,
                                                const arma::mat& lambdas,
                                                const arma::vec& alphas,
                                                double epsilon = 1e-10) {
  return PoissonEM::parallel_compute_poisson_log_posteriors(xdata, lambdas, alphas, epsilon);
}
