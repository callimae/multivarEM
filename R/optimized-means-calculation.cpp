// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <thread>  // Dla std::thread::hardware_concurrency()
#include <algorithm> // Dla std::max
using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// Wersja sekwencyjna - dobra dla mniejszych zbiorów danych
// [[Rcpp::export]]
arma::mat compute_means_simple(const arma::mat& post_plus, const arma::mat& xdata, const arma::vec& post_plus_sum) {
    // Utwórz macierz wynikową
    arma::mat means = post_plus.t() * xdata;
    
    // Dzielenie każdego wiersza przez odpowiedni element wektora sum
    for (size_t i = 0; i < means.n_rows; i++) {
        if (post_plus_sum(i) > 0) {
            means.row(i) /= post_plus_sum(i);
        }
    }
    
    return means;
}

// Funkcja do określania liczby wątków
inline unsigned int getNumThreads() {
    // Użyj liczby wątków o jeden mniejszej niż dostępna
    unsigned int hardware_threads = std::thread::hardware_concurrency();
    
    if (hardware_threads <= 1) {
        return 1; // Minimum jeden wątek
    } else {
        return hardware_threads - 1; // Pozostaw jeden wątek dla systemu
    }
}

// Struktura do obliczeń równoległych
struct ComputeMeansWorker : public Worker {
    // Dane wejściowe
    const RMatrix<double> post_plus;
    const RMatrix<double> xdata;
    const RVector<double> post_sum;
    
    // Wynik
    RMatrix<double> result;
    
    // Konstruktor
    ComputeMeansWorker(const NumericMatrix& post_plus, 
                      const NumericMatrix& xdata,
                      const NumericVector& post_sum, 
                      NumericMatrix& result)
        : post_plus(post_plus), xdata(xdata), post_sum(post_sum), result(result) {}
    
    // Operator obliczeń
    void operator()(std::size_t begin, std::size_t end) {
        const int p = result.ncol(); // liczba zmiennych (kolumn w wynikowej macierzy)
        const int n = post_plus.nrow(); // liczba obserwacji
        
        // Dla każdego klastra w przydzielonym zakresie
        for (std::size_t cluster = begin; cluster < end; cluster++) {
            // Dla każdej zmiennej
            for (int var = 0; var < p; var++) {
                double sum = 0.0;
                
                // Dla każdej obserwacji
                for (int obs = 0; obs < n; obs++) {
                    sum += post_plus(obs, cluster) * xdata(obs, var);
                }
                
                // Zapisz wynik po normalizacji
                if (post_sum[cluster] > 0) {
                    result(cluster, var) = sum / post_sum[cluster];
                } else {
                    result(cluster, var) = 0.0;
                }
            }
        }
    }
};

// Wersja równoległa - lepsza dla dużych zbiorów danych
// [[Rcpp::export]]
NumericMatrix compute_means_parallel(const NumericMatrix& post_plus, 
                                   const NumericMatrix& xdata, 
                                   const NumericVector& post_plus_sum) {
    const int k = post_plus.ncol(); // liczba klastrów
    const int p = xdata.ncol(); // liczba zmiennych
    
    // Inicjalizacja wyniku
    NumericMatrix result(k, p);
    
    // Uruchomienie obliczeń równoległych
    ComputeMeansWorker worker(post_plus, xdata, post_plus_sum, result);
    
    // Uruchom obliczenia równoległe - dopasuj wielkość ziarna do danych
    unsigned int num_threads = getNumThreads();
    // Jawne określenie typu dla std::max aby uniknąć błędów kompilacji
    int grain_size = std::max<int>(1, static_cast<int>(k / (4 * num_threads)));
    parallelFor(0, k, worker, grain_size);
    
    return result;
}

// Funkcja opakowująca, która konwertuje typy
// [[Rcpp::export]]
arma::mat compute_means_wrapper(const arma::mat& post_plus, 
                              const arma::mat& xdata, 
                              const arma::vec& post_plus_sum) {
    // Dla większości przypadków, wystarczy prosta implementacja
    if (post_plus.n_rows < 10000) {
        return compute_means_simple(post_plus, xdata, post_plus_sum);
    }
    
    // Dla dużych zbiorów, przygotuj dane do przetwarzania równoległego
    NumericMatrix post_plus_rcpp(post_plus.n_rows, post_plus.n_cols);
    NumericMatrix xdata_rcpp(xdata.n_rows, xdata.n_cols);
    NumericVector post_sum_rcpp(post_plus_sum.n_elem);
    
    // Kopiuj dane
    for (size_t i = 0; i < post_plus.n_rows; i++) {
        for (size_t j = 0; j < post_plus.n_cols; j++) {
            post_plus_rcpp(i, j) = post_plus(i, j);
        }
    }
    
    for (size_t i = 0; i < xdata.n_rows; i++) {
        for (size_t j = 0; j < xdata.n_cols; j++) {
            xdata_rcpp(i, j) = xdata(i, j);
        }
    }
    
    for (size_t i = 0; i < post_plus_sum.n_elem; i++) {
        post_sum_rcpp[i] = post_plus_sum(i);
    }
    
    // Wywołaj wersję równoległą
    NumericMatrix result = compute_means_parallel(post_plus_rcpp, xdata_rcpp, post_sum_rcpp);
    
    // Konwersja wyniku z powrotem do arma::mat
    arma::mat result_arma(post_plus.n_cols, xdata.n_cols);
    for (size_t i = 0; i < post_plus.n_cols; i++) {
        for (size_t j = 0; j < xdata.n_cols; j++) {
            result_arma(i, j) = result(i, j);
        }
    }
    
    return result_arma;
}