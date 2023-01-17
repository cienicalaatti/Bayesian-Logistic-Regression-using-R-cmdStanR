data {
  int<lower=1> D; // number of variables
  int<lower=0> N_train; // number of datapoints
  int<lower=0> N_test;
  array[N_train] int<lower=0, upper=1> y; // outcome
  array[N_train] row_vector[D] x_train; // feature vectors
  array[N_test] row_vector[D] x_pred;
}
parameters {
  vector[D] beta;
  real alpha;
}
model {
  alpha ~ cauchy(0,1);
  for (d in 1:D) {
    beta[d] ~ cauchy(0,1);
  }
  for (n in 1:N_train) {
    y[n] ~ bernoulli(inv_logit(alpha + x_train[n] * beta));
  }
}
generated quantities {
  vector[N_train] log_lik;
  for (i in 1:N_train)
    log_lik[i]=inv_logit(alpha + x_train[i] * beta);
    
  array[N_train] int y_rep;
  for (n in 1:N_train) y_rep[n] = bernoulli_logit_rng(alpha + x_train[n, ] * beta);
}
