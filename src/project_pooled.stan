data {
  int<lower=1> D; // number of variables
  int<lower=0> N_train; // number of datapoints in the training set
  int<lower=0> N_test; // number of datapoints in the testing set
  array[N_train] int<lower=0, upper=1> y; // outcome
  array[N_train] row_vector[D] x_train; // covariate matrix for the training set
  array[N_test] row_vector[D] x_pred; // covariate vector for the testing set
}
parameters {
  vector[D] beta; // regression parameters
  real alpha; // intercept
}
model {
  //priors
  alpha ~ normal(0,10);
  for (d in 1:D) {
    beta[d] ~ normal(0, 10);
  }
  //likelihood
  for (n in 1:N_train) {
    y[n] ~ bernoulli(inv_logit(alpha + x_train[n] * beta));
  }
}
generated quantities {
  //log likelihood values for the ELPD-LOO calculation
  vector[N_train] log_lik;
  for (n in 1:N_train)
    //log_lik[n]=inv_logit(alpha + x_train[n] * beta);
    log_lik[n]=bernoulli_logit_lpmf(y[n] | alpha + x_train[n] * beta);
  
  // posterior predictions for the training data
  array[N_train] int y_rep;
  for (n in 1:N_train){
    y_rep[n] = bernoulli_logit_rng(alpha + x_train[n, ] * beta); 
  }
  // posterior predictions for unseen data
  array[N_test] int y_pred;
  for (n in 1:N_test){
    y_pred[n] = bernoulli_logit_rng(alpha + x_pred[n, ] * beta); 
  }
}

