// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

struct LlmvnormWorkerOptimized : public Worker {
  // Wejście
  const RMatrix<double> xdata;
  const RVector<double> vmeans;
  const RVector<double> vvars;
  const double valph;
  const double log_term;
  const RVector<double> inv_vvars; // Precomputed inverse variances
  
  // Wynik
  RVector<double> result;
  
  // Konstruktor
  LlmvnormWorkerOptimized(const NumericMatrix& xdata, 
                         const NumericVector& vmeans, 
                         const NumericVector& vvars,
                         const NumericVector& inv_vvars,
                         double valph, 
                         double log_term, 
                         NumericVector& result)
    : xdata(xdata), vmeans(vmeans), vvars(vvars), valph(valph), 
      log_term(log_term), inv_vvars(inv_vvars), result(result) {}
  
  // Operator obliczeń
  void operator()(std::size_t begin, std::size_t end) {
    const double log_alpha = log(valph);
    const int ncol = xdata.ncol();
    
    for (std::size_t i = begin; i < end; i++) {
      double sum_sq_diff = 0.0;
      
      // Używamy prekomputowanych odwrotności wariancji zamiast dzielenia w pętli
      for (int j = 0; j < ncol; ++j) {
        const double diff = xdata(i, j) - vmeans[j];
        sum_sq_diff += (diff * diff) * inv_vvars[j];
      }
      
      result[i] = log_alpha - log_term - 0.5 * sum_sq_diff;
    }
  }
};

// [[Rcpp::export]]
NumericVector llmvnormParallelOptimized(const NumericMatrix& xdata, 
                                      const NumericVector& vmeans, 
                                      const NumericVector& vvars,
                                      double valph, 
                                      double nvar,
                                      int grain_size = 100) {
  // Inicjalizacja wektora wynikowego
  NumericVector result(xdata.nrow());
  
  // Obliczanie stałej normalizacyjnej (log_term)
  double log_term = 0.5 * sum(log(vvars)) - (nvar / 2) * log(2 * M_PI);
  
  // Prekomputacja odwrotności wariancji dla optymalizacji
  NumericVector inv_vvars(vvars.size());
  for (int j = 0; j < vvars.size(); j++) {
    inv_vvars[j] = 1.0 / vvars[j];
  }
  
  // Utworzenie i uruchomienie workera
  LlmvnormWorkerOptimized worker(xdata, vmeans, vvars, inv_vvars, valph, log_term, result);
  
  // Użyj grain_size dla lepszego balansowania obciążenia
  parallelFor(0, xdata.nrow(), worker, grain_size);
  
  return result;
}

// Funkcja zachowująca oryginalny interfejs dla kompatybilności wstecznej
// [[Rcpp::export]]
NumericVector llmvnormParallel2(const NumericMatrix& xdata, 
                               const NumericVector& vmeans, 
                               const NumericVector& vvars,
                               double valph, 
                               double nvar) {
  return llmvnormParallelOptimized(xdata, vmeans, vvars, valph, nvar);
}
