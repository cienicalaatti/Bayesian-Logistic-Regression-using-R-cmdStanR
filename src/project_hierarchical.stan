data {
  int<lower=1> D; // number of variables
  int<lower=0> N_train; // number of datapoints in the training set
  int<lower=0> N_test; // number of datapoints in the testing set
  int<lower=1> L; // number of groups
  array[N_train] int<lower=0, upper=1> y; // outcome
  array[N_train] int<lower=1, upper=L> ll; // group column for train data
  array[N_train] row_vector[D] x_train; // covariate matrix for the training set
  array[N_test] row_vector[D] x_pred; // covariate matrix for the testing set
  array[N_test] int<lower=1, upper=L> ll_pred; // group column for test data
}
parameters {
  // hyperparameters:
  array[D] real mu;
  array[D] real<lower=0> sigma;
  // regression parameters
  array[L] vector[D] beta;
  // intercepts
  array[L] real alpha;
}
model {
  for (d in 1:D) {
    mu[d] ~ normal(0, 10); // hyperprior (same for both groups)
    sigma[d] ~ inv_chi_square(10); // hyperprior (same for both groups)
    for (l in 1:L) {
      alpha[l] ~ normal(0,10); //prior, intercept
      beta[l, d] ~ normal(mu[d], sigma[d]); //prior, beta
    }
  }
  //likelihood
  for (n in 1:N_train) {
    y[n] ~ bernoulli(inv_logit(alpha[ll[n]] + x_train[n] * beta[ll[n]]));
  }
}
generated quantities {
  //log likelihood values for ELDP-LOO calculation
  vector[N_train] log_lik;
  for (n in 1:N_train)
    //log_lik[n]=inv_logit(alpha[ll[n]] + x_train[n] * beta[ll[n]]);
    log_lik[n]=bernoulli_logit_lpmf(y[n]|alpha[ll[n]] + x_train[n] * beta[ll[n]]);
  
  // posterior predictions for the training data
  array[N_train]int y_rep;
  for (n in 1:N_train){
    y_rep[n] = bernoulli_logit_rng(alpha[ll[n]] + x_train[n, ] * beta[ll][n]);
  }
  // posterior predictions for unseen data
  array[N_test] int y_pred;
  for (n in 1:N_test){
    y_pred[n] = bernoulli_logit_rng(alpha[ll[n]] + x_pred[n, ] * beta[ll_pred][n]);
  }  
}

