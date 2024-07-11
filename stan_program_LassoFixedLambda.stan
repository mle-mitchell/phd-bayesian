data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] x;   // predictor matrix
  vector[N] y;      // outcome vector
  real<lower=0> lambda; // fixed global shrinkage 
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
  vector<lower=0>[K] tausq;       //  local shrinkage
}
model {
  y ~ normal(x * beta + alpha, sigma);  // likelihood
  //beta ~ double_exponential(0,(sigma*sigma)/lambda); // hyperprior for beta
  //beta ~ double_exponential(0,sigma/lambda); // hyperprior for beta
  beta ~ normal(0,sigma*sqrt(lambda)*sqrt(tausq));
  tausq ~ exponential(1);
}