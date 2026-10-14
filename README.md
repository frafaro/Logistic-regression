# Logistic Regression for Credit Scoring

This project provides an overview of logistic regression applied to **credit scoring**: the goal is to model an individual's creditworthiness (`affidabilita`, coded as *bad*/*good*) using the other covariates in the dataset as input variables.

## Project files

| File | Description |
|------|-------------|
| `Logistic_regression.R` | Main script with the full analysis |
| `credit.RData` | Dataset with the credit information of the clients |
| `Logistic_regression.Rproj` | RStudio project file |

## Requirements

- R (with RStudio recommended)
- Packages: `ResourceSelection` (Hosmer–Lemeshow test), `ROCR` (ROC curve)

```r
install.packages(c("ResourceSelection", "ROCR"))
```

## Analysis steps

The script follows these steps:

### 1. Loading data
Load and explore the `credit` dataset (`str`, `head`, `summary`). The response variable is a factor, so the (1, 2) codification for *bad*/*good* can be kept. The data is then split into a **training set (75%)** and a **test set (25%)** with `set.seed(123)` for reproducibility.

### 2. Logistic model with 1 regressor
Fit a simple model `affidabilita ~ saldo_conto_bin` with `glm(..., family = binomial)`:
- **Output interpretation**: both intercept and regressor are significant; a positive account balance increases the logit by 1.69.
- **Odds ratio**: `exp(coef(mod1))` = 5.46 — a positive balance makes a client 5.46 times more likely to be a "good" creditor.
- **Probability estimation**: applying the logistic formula, the estimated probability of being "good" is **88.32%** with a positive balance and **58.09%** with a negative one.

### 3. Logistic model with all regressors
Fit the full model `affidabilita ~ .` on the training set. Since some parameters are not significant, a **backward stepwise regression** (`step(modT, direction = "backward")`) is run to keep only the relevant regressors, based on deviance/AIC.

### 4. Goodness of fit
- **Hosmer–Lemeshow test** (`hoslem.test`, g = 10): the null hypothesis is that the model fits the data well, so a high p-value is desired. Here p-value ≈ 0.95 → good fit.
- **Residuals**: deviance and Pearson residuals are computed with `resid()`.

### 5. ANOVA method
Compare nested models with `anova(mod1, mod2, test = "Chisq")`. The small p-value rejects the hypothesis that the two models are equivalent, so the model with 2 regressors is preferred. Alternatively, the model with the lowest AIC can be chosen.

### 6. Predictions
Use the stepwise model to predict on the test set (`predict(..., type = "response")`), obtaining the estimated probability of being "good" for each observation.

### 7. Accuracy matrix
Build a contingency table between actual and predicted values using the default **0.5 cutoff**. The model predicts "good" clients well but is less accurate on "bad" ones. Overall accuracy = **72.4%**.

### 8. ROC curve and Youden's J
Using the `ROCR` package on the training set (in-sample benchmark):
- Plot the **ROC curve** (TPR vs FPR at different thresholds).
- Compute **Youden's J statistic** (J = TPR − FPR) to find the cutoff with the best balance between sensitivity and specificity.
- Optimal cutoff = **0.6743** (J = 0.4747, sensitivity = 0.7408, specificity = 0.7339).

### 9. Final matrix evaluation
Recompute the contingency table on the test set with the optimal cutoff 0.6743. Accuracy decreases to **69.6%**, but the model identifies "bad" clients better — a desirable trade-off in credit scoring.

## How to run

1. Open `Logistic_regression.Rproj` in RStudio.
2. Load the dataset: `load("credit.RData")`.
3. Run `Logistic_regression.R` step by step.
