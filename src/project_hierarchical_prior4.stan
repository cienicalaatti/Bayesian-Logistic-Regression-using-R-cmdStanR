data {
  int<lower=1> D; // number of variables
  int<lower=0> N_train; // number of datapoints
  int<lower=0> N_test;
  int<lower=1> L; // number of groups
  array[N_train] int<lower=0, upper=1> y; // outcome
  array[N_train] int<lower=1, upper=L> ll; // group column
  array[N_train] row_vector[D] x_train; // feature vectors
  array[N_test] row_vector[D] x_pred;
}
parameters {
  array[D] real mu;
  array[D] real<lower=0> sigma;
  array[L] vector[D] beta;
  array[L] real alpha;
}
model {
  for (d in 1:D) {
    mu[d] ~ cauchy(0, 1);
    sigma[d] ~ cauchy(0,1);
    for (l in 1:L) { 
      alpha[l] ~ cauchy(0,1);
      beta[l, d] ~ normal(mu[d], sigma[d]);
    }
  }
  for (n in 1:N_train) {
    y[n] ~ bernoulli(inv_logit(alpha[ll[n]] + x_train[n] * beta[ll[n]]));
  }
}
generated quantities {
  vector[N_train] log_lik;
  for (n in 1:N_train)
    log_lik[n]=inv_logit(alpha[ll[n]] + x_train[n] * beta[ll[n]]);
    
  array[N_train] int y_rep;
    for (n in 1:N_train) y_rep[n] = bernoulli_logit_rng(alpha[ll[n]] + x_train[n, ] * beta[ll][n]);
}

