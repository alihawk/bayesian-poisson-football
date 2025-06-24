#-------------- begin hw.py --------------
"""
hw.py

This script completes a two‐part assignment on Bayesian Poisson regression:

Part 1:
  - Load and clean the football dataset.
  - Perform exploratory data analysis (EDA).
  - Build a design matrix (standardize features, add intercept).
  - Fit a Bayesian Poisson GLM via MCMC (PyMC).
  - Display MCMC diagnostics (traceplots, autocorrelation, ESS, R̂).

Part 2:
  - Implement Laplace Approximation from scratch (hand‐derived gradient & Hessian).
  - Validate Laplace fit against the MCMC posterior.
  - Use the Laplace Gaussian to make point predictions on the test set
    under three loss functions: squared‐error, absolute‐error, and 0–1 accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import csv
from sklearn.preprocessing import StandardScaler
import pymc as pm
import arviz as az
from scipy.optimize import minimize
from scipy.stats import poisson

# =============================================================================
#                                 Part 1
# =============================================================================

def load_and_clean_data(filepath):
    """
    1. Read the raw CSV file as a text string.
    2. Remove all double-quote characters (") to simplify parsing.
    3. Parse the cleaned text with pandas.read_csv(), using semicolons (;) as the delimiter.
    4. Convert all columns to numeric dtype, coercing any parse errors to NaN.
       This step ensures that any non-numeric entries become NaN.
    5. Return the resulting DataFrame, where each numeric column is float64
       and any invalid entries are NaN.

    Arguments:
    - filepath (str): Path to the 'football.csv' file.

    Returns:
    - df (pd.DataFrame): Cleaned DataFrame with numeric columns.
    """
    # Step 1: Open the file and read its contents, then remove all double quotes
    with open(filepath, "r") as f:
        text = f.read().replace('"', "")

    # Step 2: Use pandas to read the semicolon-separated values from the cleaned text
    df = pd.read_csv(io.StringIO(text), sep=";")

    # Step 3: Convert every column to numeric type (float). Non-numeric strings become NaN.
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def exploratory_data_analysis(df):
    """
    Perform a quick exploratory data analysis (EDA) on the DataFrame:
    - Print column names.
    - Print DataFrame.info() to show dtypes and non-null counts.
    - Print the number of missing values per column.
    - Print descriptive summary statistics (count, mean, std, min, max, quartiles).
    - Plot histograms for all numeric columns to visualize distribution shapes.

    Arguments:
    - df (pd.DataFrame): The cleaned DataFrame returned by load_and_clean_data().
    """
    # Print all column names as a list
    print("Columns:", df.columns.tolist(), "\n")

    # Print DataFrame info: dtypes, non-null counts, and memory usage
    df.info()

    # Print the count of missing (NaN) values in each column
    print("\nMissing per column:\n", df.isnull().sum(), "\n")

    # Print descriptive statistics for numeric columns
    # (e.g., count, mean, std, min, 25%, 50%, 75%, max)
    print("Summary:\n", df.describe(), "\n")

    # Draw histograms for all columns of numeric dtype
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols].hist(bins=15, figsize=(10, 8))
    plt.tight_layout()
    plt.show()


def build_design_matrix(df):
    """
    Build the training and test design matrices from the cleaned DataFrame:
    1. Split the DataFrame into train and test sets:
       - train_df: rows where 'GoalsScored' is NOT null (i.e., we know the target).
       - test_df: rows where 'GoalsScored' is null (i.e., 100 held-out matches).
    2. Identify predictor column names: all columns starting with "X_".
    3. Standardize the predictor values (zero mean, unit variance) using StandardScaler:
       - Fit the scaler ONLY on training predictor values.
       - Transform both training and test predictor values.
    4. Add an "Intercept" column (constant 1.0) to both standardized train and test frames.
    5. Reorder columns so "Intercept" is the first column, followed by all X_* predictors.
    6. Extract the response vector y_train from train_df, converting it to a NumPy integer array.
    7. Return:
       - X_train (pd.DataFrame): standardized predictors + intercept for training (shape 1750×9).
       - y_train (np.ndarray): integer array of length 1750 containing GoalsScored.
       - X_test  (pd.DataFrame): standardized predictors + intercept for test (shape 100×9).

    Arguments:
    - df (pd.DataFrame): The cleaned DataFrame containing both train and test rows.

    Returns:
    - X_train (pd.DataFrame): Design matrix for training features.
    - y_train (np.ndarray): Response vector (GoalsScored) for training.
    - X_test  (pd.DataFrame): Design matrix for test features.
    """
    # Step 1: Split into train/test by missingness of GoalsScored
    train_df = df[df.GoalsScored.notnull()].copy()  # 1750 rows where GoalsScored is known
    test_df  = df[df.GoalsScored.isnull()].copy()   # 100 rows where GoalsScored is missing

    # Step 2: Identify predictor column names that start with "X_"
    predictors = [c for c in df.columns if c.startswith("X_")]

    # Step 3: Standardize predictors using StandardScaler
    scaler = StandardScaler()

    # 3a. Fit on training predictors, then transform train
    X_train_std = pd.DataFrame(
        scaler.fit_transform(train_df[predictors]),
        columns=predictors,
        index=train_df.index
    )
    # 3b. Use the same scaler to transform test predictors
    X_test_std = pd.DataFrame(
        scaler.transform(test_df[predictors]),
        columns=predictors,
        index=test_df.index
    )

    # Step 4: Add an intercept column (constant term) to both
    X_train_std["Intercept"] = 1.0
    X_test_std["Intercept"]  = 1.0

    # Step 5: Reorder columns so that "Intercept" is first
    cols = ["Intercept"] + predictors
    X_train = X_train_std[cols]
    X_test  = X_test_std[cols]

    # Step 6: Extract the response vector y_train (integer GoalsScored) from train_df
    # Converting to int ensures it’s suitable for Poisson modeling.
    y_train = train_df["GoalsScored"].astype(int).values

    # Return the constructed design matrices and response
    return X_train, y_train, X_test


def fit_bayesian_poisson(X_train, y_train):
    """
    Fit a Bayesian Poisson Generalized Linear Model (GLM) using PyMC:
    - Model: y_i ~ Poisson(λ_i), where λ_i = exp(η_i) and η_i = X_i · β.
    - Priors: β_j ~ Normal(0, 10^2) independently for j=0..8 (including intercept).
    - Sampling: Use NUTS (No-U-Turn Sampler) with 4 chains, 1000 tuning steps, 2000 draws.

    Arguments:
    - X_train (pd.DataFrame): Training design matrix (1750×9).
    - y_train (np.ndarray): Training response (length 1750).

    Returns:
    - trace (arviz.InferenceData): The MCMC trace containing posterior draws of β.
    """
    # Convert X_train DataFrame to a NumPy array, shape (1750, 9)
    X = X_train.values

    # Build the PyMC model in a context manager
    with pm.Model() as model:
        # 1. Define a vector of 9 independent Normal priors for β (mean=0, sd=10)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0, shape=X.shape[1])

        # 2. Linear predictor η = X · β (vectorized)
        eta = pm.math.dot(X, beta)  # shape (1750,)

        # 3. Log-link: λ = exp(η), so λ_i > 0
        lam = pm.math.exp(eta)      # shape (1750,)

        # 4. Observed data: y_train ~ Poisson(μ=λ)
        y_obs = pm.Poisson("y_obs", mu=lam, observed=y_train)

        # 5. Sample from the posterior using NUTS
        trace = pm.sample(
            draws=2000,         # Number of post‐tuning samples per chain
            tune=1000,          # Number of warm‐up (tuning) steps per chain
            chains=4,           # Number of independent MCMC chains
            target_accept=0.9,  # Increase acceptance probability to reduce divergences
            random_seed=42      # For reproducibility
        )

    # Return the MCMC trace to inspect or summarize later
    return trace


def run_mcmc_diagnostics(trace):
    """
    Generate standard MCMC diagnostics for the fitted Poisson model:
    - Traceplot for each β_j to visually check mixing across chains.
    - Autocorrelation plot for β to check how quickly samples become independent.
    - Numeric summary including:
        * mean: posterior mean of each β_j
        * sd:   posterior standard deviation of each β_j
        * hdi_3%, hdi_97%: 94% highest density interval (credible interval)
        * ess_bulk: effective sample size in the bulk (measures sample independence)
        * ess_tail: effective sample size in the tails (important for quantiles)
        * r_hat: Gelman‐Rubin statistic (should be ~1.00 if chains converged)

    Arguments:
    - trace (arviz.InferenceData): The MCMC trace from fit_bayesian_poisson().

    Returns:
    - summary (pd.DataFrame): The ArviZ summary table of posterior statistics.
    """
    # Plot tracelines + posterior densities for each β_j
    az.plot_trace(trace)

    # Plot autocorrelation for β_j to check thinning / independence
    az.plot_autocorr(trace, var_names=["beta"])

    # Print a numeric summary table of posterior draws for β
    summary = az.summary(trace, var_names=["beta"], round_to=2)
    print(summary)

    # Return the summary DataFrame for any further use
    return summary


# =============================================================================
#                                 Part 2
# =============================================================================

def log_posterior(beta, X, y, sigma=10.0):
    """
    Compute the log-posterior ℓ(β) = log-likelihood + log-prior for the Poisson model:
      log-likelihood:   ∑[ y_i * (x_i·β)  −  exp(x_i·β ) ]
      Gaussian prior:   −(1 / (2σ²)) ∑ β_j²   (ignoring constant term in prior)

    Arguments:
    - beta (np.ndarray, shape (d,)): Coefficient vector of length d (d=9 here).
    - X    (np.ndarray, shape (N, d)): Design matrix for N training examples.
    - y    (np.ndarray, shape (N,)):  Observed counts (GoalsScored).
    - sigma (float): Prior standard deviation for each β_j (default 10.0).

    Returns:
    - float: The log-posterior value at β.
    """
    # 1. Linear predictor: η = X @ β  → shape (N,)
    eta = X.dot(beta)

    # 2. Poisson rate: λ_i = exp(η_i)  → shape (N,)
    lam = np.exp(eta)

    # 3. Poisson log-likelihood: ∑ [y_i * η_i − λ_i]
    ll = np.sum(y * eta - lam)

    # 4. Gaussian log-prior: −(1/(2σ²)) ∑ β_j²
    lp = -0.5 * np.sum(beta**2) / (sigma**2)

    # 5. Sum: ℓ(β) = log-likelihood + log-prior (constant offset dropped)
    return ll + lp


def grad_log_posterior(beta, X, y, sigma=10.0):
    """
    Compute the gradient ∇ℓ(β), which is a length-d vector of partial derivatives:
      ∇ℓ(β) = X^T (y - λ)  −  (1/σ²) β,
      where λ = exp(Xβ).

    Arguments:
    - beta  (np.ndarray, shape (d,)): Current coefficient vector.
    - X     (np.ndarray, shape (N, d)): Design matrix.
    - y     (np.ndarray, shape (N,)): Observed counts.
    - sigma (float): Prior standard deviation.

    Returns:
    - grad (np.ndarray, shape (d,)): The gradient ∂ℓ/∂β at the given β.
    """
    # 1. Compute linear predictor and rates
    eta = X.dot(beta)          # shape (N,)
    lam = np.exp(eta)          # shape (N,)

    # 2. Residual: (y_i - λ_i) for each i
    residual = y - lam         # shape (N,)

    # 3. Likelihood part of gradient: X^T @ (y - λ)  → shape (d,)
    grad_ll = X.T.dot(residual)

    # 4. Prior part of gradient: derivative of −(1/(2σ²)) ∑ β_j² is −β/σ²
    grad_lp = -beta / (sigma**2)

    # 5. Sum components: ∇ℓ = grad_ll + grad_lp
    return grad_ll + grad_lp


def hessian_log_posterior(beta, X, y, sigma=10.0):
    """
    Compute the Hessian ∇²ℓ(β), a (d×d) matrix of second derivatives:
      ∇²ℓ(β) = − X^T diag(λ) X  −  (1/σ²) I_d,
      where λ_i = exp(x_i·β).

    We implement this efficiently by noting that:
      X^T diag(λ) X = (X_scaled)^T @ X_scaled,
    where X_scaled has row i = sqrt(λ_i) * x_i.

    Arguments:
    - beta  (np.ndarray, shape (d,)): Current coefficient vector.
    - X     (np.ndarray, shape (N, d)): Design matrix.
    - y     (np.ndarray, shape (N,)): Observed counts.
    - sigma (float): Prior standard deviation.

    Returns:
    - H (np.ndarray, shape (d, d)): The Hessian matrix at β.
    """
    # 1. Compute linear predictor and rates
    eta = X.dot(beta)         # shape (N,)
    lam = np.exp(eta)         # shape (N,)

    # 2. Compute sqrt(λ) for each data point
    sqrt_lam = np.sqrt(lam)   # shape (N,)

    # 3. Scale each row of X by sqrt_lam[i] to form X_scaled
    #    X_scaled[i, :] = X[i, :] * sqrt_lam[i]
    X_scaled = X * sqrt_lam[:, None]  # resulting shape (N, d)

    # 4. Likelihood part of Hessian: −(X_scaled)^T @ (X_scaled)
    H_ll = - X_scaled.T.dot(X_scaled)  # shape (d, d)

    # 5. Prior part of Hessian: −(1/σ²) * I_d
    H_lp = -np.eye(X.shape[1]) / (sigma**2)  # shape (d, d)

    # 6. Sum: ∇²ℓ = H_ll + H_lp
    return H_ll + H_lp


def fit_laplace(X_train, y_train, sigma=10.0):
    """
    Perform the Laplace Approximation to approximate the posterior p(β|y)
    by a Gaussian centered at the MAP (β_MAP) with covariance Σ_Laplace = [−H(β_MAP)]⁻¹.

    Steps:
      1. Convert X_train, y_train to NumPy arrays.
      2. Define objective = −log_posterior(β) because scipy.optimize.minimize minimizes.
      3. Provide gradient = −grad_log_posterior(β) and Hessian = −hessian_log_posterior(β).
      4. Call minimize() with method="trust-constr" for a Newton‐style solver.
      5. Extract β_MAP = res.x and check for success.
      6. Compute H_MAP = Hessian at β_MAP, then Σ_Laplace = inv(−H_MAP).

    Arguments:
    - X_train (pd.DataFrame): Training design matrix (1750×9).
    - y_train (np.ndarray): Training response (length 1750).
    - sigma   (float): Prior standard deviation (default 10.0).

    Returns:
    - beta_map (np.ndarray, shape (d,)): The MAP estimate of β.
    - Sigma_laplace (np.ndarray, shape (d, d)): Covariance matrix of the Laplace Gaussian.
    """
    # Convert to NumPy arrays for numeric operations
    X = X_train.values    # shape (1750, 9)
    y = y_train           # shape (1750,)

    # Initial guess for β: all zeros (length d=9)
    beta0 = np.zeros(X.shape[1])

    # Define the objective function (−log_posterior)
    def objective(beta):
        return -log_posterior(beta, X, y, sigma)

    # Define the gradient of the objective (−grad_log_posterior)
    def objective_grad(beta):
        return -grad_log_posterior(beta, X, y, sigma)

    # Define the Hessian of the objective (−hessian_log_posterior)
    def objective_hess(beta):
        return -hessian_log_posterior(beta, X, y, sigma)

    # Call scipy.optimize.minimize with a Newton‐like solver that uses Hessian
    res = minimize(
        fun=objective,
        x0=beta0,
        jac=objective_grad,
        hess=objective_hess,
        method="trust-constr",           # uses Hessian for Newton steps
        options={
            "gtol": 1e-8,                # gradient tolerance for convergence
            "xtol": 1e-8,                # parameter tolerance for convergence
            "maxiter": 100               # maximum number of iterations
        }
    )

    # If optimization failed, raise an error with the solver’s message
    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)

    # Extract the optimized coefficients: β_MAP
    beta_map = res.x  # shape (9,)

    # Compute the Hessian at β_MAP
    H_map = hessian_log_posterior(beta_map, X, y, sigma)  # shape (9, 9)

    # Compute Σ_Laplace = inverse of (−H_map)
    Sigma_laplace = np.linalg.inv(-H_map)  # shape (9, 9)

    return beta_map, Sigma_laplace


def validate_laplace_vs_mcmc(beta_map, Sigma_laplace, trace, X_train):
    """
    Compare the Laplace approximation's mean & standard deviation per coefficient
    to the MCMC posterior mean & standard deviation from Part 1.

    Steps:
      1. Extract MCMC summary (means and sds) for each β_j from the PyMC trace.
      2. Compute Laplace means = β_MAP and Laplace sds = sqrt(diag(Σ_Laplace)).
      3. Create a pandas DataFrame with columns:
         ['param', 'mcmc_mean', 'mcmc_sd', 'lap_mean', 'lap_sd'].
      4. Print the DataFrame so you can verify each row matches closely.

    Arguments:
    - beta_map      (np.ndarray, shape (d,)): MAP estimate from fit_laplace().
    - Sigma_laplace (np.ndarray, shape (d, d)): Laplace covariance from fit_laplace().
    - trace         (arviz.InferenceData): MCMC trace from Part 1.
    - X_train       (pd.DataFrame): Training design matrix to retrieve column names.
    """
    # 1. Get MCMC summary for the "beta" variable
    mcmc_summary = az.summary(trace, var_names=["beta"], round_to=2)

    # 2. Extract MCMC posterior means and sds (arrays of length d=9)
    mcmc_means = mcmc_summary["mean"].values
    mcmc_sds   = mcmc_summary["sd"].values

    # 3. Laplace means = β_MAP
    lap_means = beta_map

    # 4. Laplace sds = sqrt of the diagonal entries of Σ_Laplace
    lap_sds = np.sqrt(np.diag(Sigma_laplace))

    # 5. Build a comparison table
    df_compare = pd.DataFrame({
        "param": ["Intercept"] + list(X_train.columns[1:]),  # param names in order
        "mcmc_mean": mcmc_means,
        "mcmc_sd":   mcmc_sds,
        "lap_mean":  lap_means,
        "lap_sd":    lap_sds
    })

    # 6. Print the comparison DataFrame
    print(df_compare)


def make_predictions_laplace(beta_map, Sigma_laplace, X_test, S=5000, K_max=10):
    """
    Use the Laplace Gaussian approximation to produce point predictions on X_test
    under three different loss functions:

      1. Squared-Error (MSE): predict the posterior mean of Y, i.e. E[Y] = E[λ].
      2. Absolute-Error (MAE): predict the posterior median of Y (approximate median of λ).
      3. 0–1 Accuracy (Exact-Count): predict the integer k that maximizes
         the posterior predictive probability P(Y = k).

    Procedure:
      A. Draw S random samples of β ~ N(β_MAP, Σ_Laplace).   → (S, d) array of betas.
      B. For each sample β^(s), compute λ^(s)_i = exp(X_test[i] · β^(s)) for all i.
         That yields an (S, N_test) matrix called "lams".
      C. MSE prediction for match i:  mean(lams[:, i]).
      D. MAE prediction for match i:  median(lams[:, i]).
      E. Mode prediction for match i:  argmax_{k in {0..K_max}} average of
         PoissonPMF(k; μ = lams[:, i]) across the S samples.

    Arguments:
    - beta_map      (np.ndarray, shape (d,)): MAP estimate of β from fit_laplace().
    - Sigma_laplace (np.ndarray, shape (d, d)): Covariance matrix from fit_laplace().
    - X_test        (pd.DataFrame, shape (N_test, d)): Test design matrix (100×9).
    - S             (int): Number of Laplace‐Gaussian draws to approximate the posterior (default 5000).
    - K_max         (int): Maximum integer outcome considered for mode calculation (default 10).

    Returns:
    - y_pred_mse  (np.ndarray, shape (N_test,)): Predictions under squared‐error loss.
    - y_pred_mae  (np.ndarray, shape (N_test,)): Predictions under absolute‐error loss.
    - y_pred_mode (np.ndarray, shape (N_test,)): Integer predictions under 0–1 loss.
    """
    # Convert X_test DataFrame to NumPy array for fast dot products
    X_test_arr = X_test.values  # shape (100, 9)

    # ----------------------------------------------------------------------------
    # A. Draw S samples from N(β_MAP, Σ_Laplace)
    # ----------------------------------------------------------------------------
    beta_samples = np.random.multivariate_normal(
        mean=beta_map,       # center of the Gaussian
        cov=Sigma_laplace,   # covariance
        size=S               # how many draws
    )  # array shape: (S, 9)

    # ----------------------------------------------------------------------------
    # B. Compute Poisson rates: λ^(s)_i = exp(X_test[i] · β^(s))
    #    We do this in a single vectorized operation:
    #    1. Compute etas = beta_samples @ X_test_arr^T → shape (S, 100)
    #    2. Then lam = exp(etas) → (S, 100) of positive rates
    # ----------------------------------------------------------------------------
    etas = beta_samples.dot(X_test_arr.T)  # shape (S, 100)
    lams = np.exp(etas)                   # shape (S, 100)

    # ----------------------------------------------------------------------------
    # C. Squared‐Error (MSE) predictions:
    #    For each test match i, we want the posterior mean of Y_i,
    #    which is E[Y_i | data] = E[λ_i].  We approximate that by:
    #      y_pred_mse[i] = mean( lams[:, i] ).
    # ----------------------------------------------------------------------------
    y_pred_mse = np.mean(lams, axis=0)  # shape (100,)

    # ----------------------------------------------------------------------------
    # D. Absolute‐Error (MAE) predictions:
    #    The Bayes‐optimal predictor for absolute error is the median of Y_i.
    #    Since Y_i ~ Poisson(λ_i) and E[Poisson(λ_i)] = λ_i, a common approximation is:
    #      y_pred_mae[i] = median( lams[:, i] ).
    #    (Alternatively, one could sample Y_i ~ Poisson(lams[:, i]) and take the median of those
    #     S draws, but taking median of λs is simpler and often quite close.)
    # ----------------------------------------------------------------------------
    y_pred_mae = np.median(lams, axis=0)  # shape (100,)

    # ----------------------------------------------------------------------------
    # E. 0–1 Accuracy (Exact‐Count) predictions:
    #    For each test match i (i = 0..99), we want to find the integer k ≥ 0 that maximizes:
    #      P(Y_i = k | data) = ∫ P(Y_i = k | β) p(β | data) dβ
    #    We approximate that integral by Monte Carlo:
    #      For each sample s, P(Y_i = k | β^(s)) = PoissonPMF(k; μ = lams[s, i]).
    #    Then we average over s = 1..S, and pick the k that has the highest average probability.
    # ----------------------------------------------------------------------------
    N_test = X_test_arr.shape[0]       # 100 matches
    y_pred_mode = np.zeros(N_test, dtype=int)

    # Loop over each test match i
    for i in range(N_test):
        lam_i = lams[:, i]                  # (S,) array of Poisson rates for match i
        avg_probs = np.empty(K_max + 1)     # will hold avg probability for k = 0..K_max

        # For each candidate integer k in [0..K_max], compute:
        #   average_{s=1..S}[ PoissonPMF(k; λ^(s)_i ) ]
        for k in range(K_max + 1):
            # poisson.pmf(k, mu=lam_i) returns (S,) array:
            #   [P(Y=k | λ^(1)), P(Y=k | λ^(2)), ..., P(Y=k | λ^(S))]
            pmf_vals = poisson.pmf(k, mu=lam_i)  # shape (S,)
            avg_probs[k] = np.mean(pmf_vals)

        # Select k that maximizes the average probability
        y_pred_mode[i] = np.argmax(avg_probs)

    # Return the three types of predictions, each of length N_test=100
    return y_pred_mse, y_pred_mae, y_pred_mode


# =============================================================================
#                                Main Execution
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Part 1: Data preparation, Bayesian Poisson MCMC, diagnostics
    # -------------------------------------------------------------------------

    # Step 1: Load and clean the raw CSV file
    df = load_and_clean_data("../data/football.csv")


    # Step 2: Perform exploratory data analysis on the cleaned DataFrame
    exploratory_data_analysis(df)

    # Step 3: Build the design matrix:
    #         X_train: DataFrame (1750 × 9), y_train: ndarray (1750,), X_test: DataFrame (100 × 9)
    X_train, y_train, X_test = build_design_matrix(df)

    # Print shapes to verify
    print("X_train shape:", X_train.shape)  # expects (1750, 9)
    print("y_train shape:", y_train.shape)  # expects (1750,)
    print("X_test shape: ", X_test.shape)   # expects (100, 9)

    # Step 4: Fit the Bayesian Poisson GLM using MCMC (PyMC). Returns an ArviZ trace
    trace = fit_bayesian_poisson(X_train, y_train)

    # Step 5: Run MCMC diagnostics (traceplot, autocorr, summary) and display summary table
    summary_mcmc = run_mcmc_diagnostics(trace)

    # -------------------------------------------------------------------------
    # Part 2: Laplace Approximation, Validation, and Prediction
    # -------------------------------------------------------------------------

    # Step 6: Fit the Laplace approximation (hand‐derived gradient & Hessian)
    beta_map, Sigma_laplace = fit_laplace(X_train, y_train, sigma=10.0)

    # Print MAP estimate and its approximate standard deviations
    print("β_MAP =", beta_map)
    lap_sds = np.sqrt(np.diag(Sigma_laplace))
    print("Laplace std devs of β:", lap_sds)

    # Step 7: Validate Laplace approximation against the MCMC posterior
    validate_laplace_vs_mcmc(beta_map, Sigma_laplace, trace, X_train)

    # Step 8: Use the Laplace Gaussian to make predictions on the 100 test rows
    y_pred_mse, y_pred_mae, y_pred_mode = make_predictions_laplace(
        beta_map, Sigma_laplace, X_test, S=5000, K_max=10
    )

    # Step 9: Package the three sets of predictions into a DataFrame and display the first 10
    predictions_df = pd.DataFrame({
        "MSE_pred":  y_pred_mse,
        "MAE_pred":  y_pred_mae,
        "Mode_pred": y_pred_mode
    })
    print("\nFirst 10 test‐set predictions:\n", predictions_df.head(10))

#-------------- end hw.py  --------------


