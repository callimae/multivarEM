// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::mat calculate_vars(const arma::mat& post_plus, const arma::mat& xdata, 
                         const arma::mat& means, arma::mat& varmin, int k) {
  
  // Sprawdzenie wymiarów wejściowych
  if (post_plus.n_cols < k) {
    Rcpp::stop("post_plus ma zbyt mało kolumn");
  }
  if (means.n_rows != k) {
    Rcpp::stop("means ma nieprawidłową liczbę wierszy");
  }
  if (means.n_cols != xdata.n_cols) {
    Rcpp::stop("means i xdata mają niezgodne wymiary");
  }
  if (varmin.n_rows != k || varmin.n_cols != xdata.n_cols) {
    Rcpp::stop("varmin ma nieprawidłowe wymiary");
  }
  
  // Inicjalizacja macierzy wariancji o odpowiednich wymiarach
  arma::mat vars(k, xdata.n_cols);
  
  // Oblicz globalną wariancję dla każdej kolumny jako fallback dla przypadków zerowej wariancji
  arma::vec global_vars(xdata.n_cols);
  for (size_t d = 0; d < xdata.n_cols; d++) {
    global_vars(d) = arma::var(xdata.col(d));
    // Jeśli globalna wariancja jest zbyt mała, ustaw jakąś minimalną wartość
    if (global_vars(d) < 1e-10) {
      global_vars(d) = 1e-6; // Minimalna wartość wariancji
    }
  }
  
  // Implementacja pętli for z kodu R, ale bez używania txdata
  for (int i = 0; i < k; i++) {
    // Pobierz kolumnę post_plus dla i-tego komponentu
    arma::vec post_i = post_plus.col(i);
    
    // Suma prawdopodobieństw posteriori dla i-tego komponentu
    double post_sum = arma::sum(post_i);
    
    // Dla każdego wymiaru (kolumny xdata)
    for (size_t d = 0; d < xdata.n_cols; d++) {
      // Inicjalizacja xvar dla tego wymiaru
      double xvar = 0.0;
      
      // Implementacja tcrossprod bez używania txdata
      // W oryginalnym kodzie: xvar <- tcrossprod(post_plus[,i], ((txdata - means[i,])^2))
      for (size_t j = 0; j < xdata.n_rows; j++) {
        // Oblicz (xdata[j,d] - means[i,d])^2
        double diff = xdata(j, d) - means(i, d);
        double diff_sq = diff * diff;
        
        // Mnóż przez post_plus[j,i] i dodaj do xvar
        xvar += post_i(j) * diff_sq;
      }
      
      // Podziel xvar przez sumę post_plus[,i]
      if (post_sum > 0) {
        vars(i, d) = xvar / post_sum;
      } else {
        // Jeśli post_sum jest zbyt małe, użyj globalnej wariancji
        vars(i, d) = global_vars(d);
      }
      
      // Sprawdź czy wariancja jest zbyt mała i użyj globalnej wariancji lub minimalnej wartości
      if (vars(i, d) < 1e-10) {
        // Najpierw spróbuj użyć varmin
        if (varmin(i, d) > 1e-10) {
          vars(i, d) = varmin(i, d);
        } 
        // Jeśli varmin też jest zbyt mała, użyj globalnej wariancji
        else if (global_vars(d) > 1e-10) {
          vars(i, d) = global_vars(d);
        }
        // Jeśli globalna wariancja też jest zbyt mała, użyj stałej minimalnej wartości
        else {
          vars(i, d) = 1e-6;
        }
      }
    }
  }
  
  // Aktualizuj varmin tylko jeśli nowe wariancje są większe od minimalnej wartości
  for (int i = 0; i < k; i++) {
    for (size_t d = 0; d < xdata.n_cols; d++) {
      // Tylko jeśli nowa wariancja jest sensowną wartością, uaktualnij varmin
      if (vars(i, d) > 1e-10) {
        varmin(i, d) = vars(i, d);
      }
    }
  }
  
  return vars;
}