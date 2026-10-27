# ML Classification Pipeline

An R-based machine learning pipeline for binary classification. It compares Linear SVM, Radial SVM, Elastic Net Logistic Regression and Random Forest, with permutation importance, forward feature selection, model tuning and interpretability analysis.

Works on any dataset, as long as the first column is the outcome and the remaining columns are the features.

## Overview

This pipeline performs end-to-end machine learning analysis including:

- Stratified train/test split
- Permutation importance for each model type
- Forward feature selection using the permutation importance ranking
- Model comparison (Linear SVM vs Radial SVM vs Elastic Net vs Random Forest)
- Final model tuning on the selected features
- SHAP-based model interpretability
- Held-out test set evaluation and reporting

## Features

### Data Checks
- Checks the outcome has exactly 2 values and works out which one is the positive class
- Stops if there are non-numeric features or missing values
- Drops constant features
- Cleans up feature names so they work in model formulas, but keeps the original names for plots and outputs

### Permutation Importance
- Repeated stratified k-fold cross-validation
- Each feature is shuffled in the held-out fold and the drop in PR-AUC is recorded
- Features ranked by the 10% trimmed mean PR-AUC drop across all folds and repeats
- Run in parallel

### Feature Selection
- Forward feature selection, adding features in order of permutation importance
- Cross-validated PR-AUC and ROC-AUC at each step, with 95% confidence intervals
- Panel size capped at one feature per 10 training samples
- Optional 1-SE rule, which picks the smallest panel within 1 SE of the best score

### Model Comparison
- Linear SVM, Radial SVM, Elastic Net and Random Forest compared on cross-validated PR-AUC
- Automated optimal model and panel size selection
- Every panel is also checked on the held-out test set (reported only, not used to pick the model)

### Final Model
- Full hyperparameter tuning on the selected panel
- Youden threshold taken from the training out-of-fold predictions, not the test set

### Model Interpretability
- SHAP (SHapley Additive exPlanations) analysis
- Permutation importance plots
- Decision boundary projection in PCA space

### Reporting
- ROC curve with bootstrap confidence intervals
- Precision-recall curve with bootstrap confidence intervals
- Confusion matrix
- Sensitivity, specificity, PPV, NPV, balanced accuracy and Brier score
- Publication-ready TIFF outputs

## Requirements

### Required R Packages
```r
library(caret)          # Machine learning framework
library(ggplot2)        # Plotting
library(ggrepel)        # Text repelling in plots
library(dplyr)          # Data manipulation
library(tidyr)          # Data reshaping
library(foreach)        # Parallel loops
library(doParallel)     # Parallel backend
library(writexl)        # Excel writing
library(readxl)         # Excel reading

# Also needed (called directly with ::)
# kernlab, e1071        SVM kernels
# randomForest          Random forest
# glmnet                Elastic net
# pROC                  ROC analysis
# PRROC                 Precision-recall curves
# MLmetrics             PR-AUC during cross-validation
# iml                   SHAP values
```

Optional: `pushoverr` for phone notifications while it runs.

## Usage

### 1. Data Preparation

**Data Requirements:**

- First column: binary outcome variable (e.g. Yes/No, 1/0)
- Remaining columns: numeric predictor variables
- No ID columns
- No missing values

The outcome gets recoded to "Case"/"Control" inside the pipeline. If the outcome is Yes/No, 1/0, TRUE/FALSE or Case/Control it picks the positive class automatically, otherwise set `positive_class` in the config.

### 2. Configuration

Everything you need to change is in the `config` list at the top of the script:

```r
config <- list(
  data_file      = "example_data.csv",  # .csv or .xlsx
  positive_class = NULL,                # e.g. "Yes", NULL = auto-detect
  strata         = NULL,                # optional, stratify by something other than the outcome
  run_name       = "Run1",              # added to every output file name
  out_dir        = "ML_Results",

  seed           = 1001,
  train_prop     = 0.70,                # 70/30 train/test split
  cv_k           = 5,                   # cross-validation folds
  perm_repeats   = 20,                  # repeats for permutation importance
  n_perm         = 5,                   # shuffles per feature per fold
  max_features   = NULL,                # NULL = 1 feature per 10 training samples
  use_one_se     = TRUE,
  ...
)
```

You can also load your data into R yourself and call it `input_data` before running the script, and it'll use that instead of `data_file`.

If you want to stratify by something other than the outcome (e.g. event type), pass a vector with one value per row to `strata`.

### 3. Run Analysis

```r
source("ML_Pipeline.R")
```

The pipeline runs automatically through several stages:

**Stage 1: Data Preprocessing**
- Data checks
- Stratified train/test split
- Stratified cross-validation folds on the training data

**Stage 2: Permutation Importance and Forward Feature Selection**
- Permutation importance for each model type
- Forward feature selection along each ranking
- Results saved after each model

**Stage 3: Model Selection**
- Performance comparison across all models
- Optimal model and panel size selection
- Held-out test check of every panel

**Stage 4: Final Model**
- Hyperparameter tuning on the optimal features
- Youden threshold
- Decision boundary and SHAP analysis
- Test set evaluation

Once Stage 2 has been run once, set `run_selection = FALSE` to reload the saved results and skip straight to the plots and final model.

## Example Data

`example_data.csv` is simulated data (240 samples, 16 features, Yes/No outcome) so you can test the pipeline before running it on your own data.

## Output Files

All outputs are saved to the `out_dir` folder (`ML_Results/` by default), with `run_name` added to the end of each file name.

### Statistical Reports
- `<Model> Permutation Importance.xlsx` - Permutation importance ranking for each model
- `<Model> Ranked FFS PRAUC.xlsx` - Forward feature selection results for each model
- `Ranked FFS Model Summary PRAUC.xlsx` - All models compared
- `Ranked FFS Panel Test PRAUC.xlsx` - Cross-validated vs held-out test performance for every panel
- `Final Model Test Metrics.xlsx` - Final model performance on the test set

### Visualizations
- `Permutation Importance.tiff` - Top 15 features for each model
- `All Comparisons Ranked FFS PRAUC.tiff` - Feature selection comparison plot
- `Ranked FFS CV vs Test PRAUC.tiff` - Cross-validated vs test performance
- `Decision Boundary.tiff` - Final model decision boundary in PCA space
- `SHapley Additive exPlanations.tiff` - SHAP summary plot
- `Confusion Matrix Youden.tiff` - Final model confusion matrix
- `Optimal Model ROC Curve.tiff` - Final model ROC curve
- `Optimal Model PR Curve.tiff` - Final model precision-recall curve

### Model Files
- `Optimised Final Model.rds` - Trained model object for future predictions

### Other
- PCA and decision boundary grid CSVs (2D and 3D), for replotting
- `parallel_log.txt` - Log from the parallel workers

## Model Details

### Cross-Validation Settings
- 5-fold stratified cross-validation (change with `cv_k`)
- The same folds are used for every model, so the comparison is fair
- Metric: PR-AUC for model selection, ROC-AUC also reported

### Model Configurations
During feature selection the hyperparameters are fixed so that only the features change:

- **Linear SVM:** `svmLinear2`, cost = 1
- **Radial SVM:** `svmRadialSigma`, sigma = 1/number of features, C = 1
- **Elastic Net:** `glmnet`, alpha = 0.5, lambda = 0.01 (plain logistic regression for a single feature)
- **Random Forest:** `rf`, mtry = sqrt(number of features), 500 trees

The final model is then fully tuned on the selected panel, using the 1-SE rule to pick the simplest setting.

**Preprocessing:** Centre and scale normalization (not needed for Random Forest)

### Performance Evaluation
- Cross-validated PR-AUC and ROC-AUC with 95% confidence intervals
- Bootstrap confidence intervals on the test set (5000 iterations)
- AUC, sensitivity, specificity, PPV, NPV, balanced accuracy, Brier score

### Adding a Model
Add it to `models` in the config, then add a fixed grid to `fixed_grid()` and a tuning grid to `tuning_grid()`.

## Troubleshooting

Any issues found please report and forward to mclarnon-t1@ulster.ac.uk

**"Can't tell which outcome value is the positive class":**
- Set `positive_class` in the config, e.g. `positive_class = "Dead"`

**"All feature columns must be numeric":**
- Remove any ID columns
- Dummy code any categorical variables

**"Missing values found":**
- Impute or remove missing values before running

**Memory errors with large datasets:**
- Reduce `n_cores`
- Reduce `perm_repeats`
- Consider data subsampling

## Computational Complexity

- Permutation importance is the slowest part: `cv_k` x `perm_repeats` model fits per model, plus `n_perm` shuffles per feature in each fold
- Random Forest is the slowest model
- SHAP analysis: ~1-2 minutes per observation

### Optimization Tips
- Reduce `perm_repeats` and `n_perm` for a quick test run (e.g. 2 and 2)
- Set `shap_max_obs` to run SHAP on a sample of the training data (e.g. 100)
- Set `run_shap = FALSE` to skip SHAP altogether
- Use `run_selection = FALSE` to reuse saved results instead of rerunning Stage 2

## License

This code is provided as-is for research and educational purposes. Please ensure you have appropriate licenses for all required R packages.
