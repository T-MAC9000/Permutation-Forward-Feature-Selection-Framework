# Permutation-ranked based FFS pipeline (binary classification)
# Input: column 1 = binary outcome, remaining columns = numeric features; no IDs, no NAs

##### Libraries #####
suppressPackageStartupMessages({
  library(caret)
  library(ggplot2)
  library(ggrepel)
  library(dplyr)
  library(tidyr)
  library(foreach)
  library(doParallel)
  library(writexl)
  library(readxl)
})
# Also required (called with ::): kernlab, e1071, randomForest, glmnet, pROC,
# PRROC, MLmetrics, iml. Optional: pushoverr (phone notifications).

##### Config #####
config <- list(
  data_file      = "example_data.csv",  # .csv / .xlsx; column 1 = outcome, rest = features
  positive_class = NULL,     # outcome value counted as "Case"; NULL = auto-detect (Yes / 1 / TRUE / Case)
  strata         = NULL,     # optional vector (one per row) to stratify split + folds; NULL = outcome
  run_name       = "Run1",   # appended to every output file name
  out_dir        = "ML_Results",
  
  seed           = 1001,
  train_prop     = 0.70,
  cv_k           = 5,        # CV folds for selection and tuning
  perm_repeats   = 20,       # repeats of stratified k-fold CV for permutation importance
  n_perm         = 5,        # shuffles per feature per held-out fold
  max_features   = NULL,     # cap on panel size; NULL = 1 feature per 10 training rows
  use_one_se     = TRUE,     # smallest panel within 1 SE of the best model's peak CV PR-AUC
  
  models = c("Linear SVM"          = "svmLinear2",
             "Radial SVM"          = "svmRadialSigma",
             "Logistic Regression" = "glmnet",
             "Random Forest"       = "rf"),
  
  run_selection  = TRUE,     # FALSE = reload saved importance / selection results from out_dir
  run_shap       = TRUE,
  shap_max_obs   = NULL,     # SHAP is slow; NULL = all training rows, or e.g. 100
  n_cores        = max(1, parallel::detectCores() - 1),
  notify         = FALSE     # TRUE = pushover notifications (needs pushoverr set up)
)

##### Functions #####

notify <- function(msg) {
  message(msg)
  if (isTRUE(config[["notify"]]) && requireNamespace("pushoverr", quietly = TRUE)) {
    try(pushoverr::pushover(message = msg), silent = TRUE)
  }
}

out_file <- function(name, ext = ".xlsx") {
  file.path(config[["out_dir"]], paste0(name, "_", config[["run_name"]], ext))
}

save_plot <- function(plot, name, width = 3000, height = 2500) {
  print(plot)
  ggsave(out_file(name, ".tiff"), plot = plot, width = width, height = height,
         units = "px", dpi = 350, limitsize = FALSE)
}

read_input <- function(path) {
  ext <- tolower(tools::file_ext(path))
  df <- switch(ext,
               csv  = read.csv(path, check.names = FALSE, stringsAsFactors = FALSE),
               xlsx = as.data.frame(readxl::read_xlsx(path)),
               xls  = as.data.frame(readxl::read_xls(path)),
               stop("Unsupported file type: .", ext, " (use .csv or .xlsx)"))
  df
}

# Validates the layout and returns:
#   data   - data frame with column "Outcome" (factor: Case, Control) + features
#            (feature names made syntactically safe for formulas)
#   labels - named vector mapping safe feature names back to the originals
prepare_data <- function(df, positive_class = NULL) {
  df <- as.data.frame(df)
  if (ncol(df) < 2) stop("Data needs an outcome column plus at least one feature column.")
  
  y_raw  <- as.character(df[[1]])
  levels_found <- sort(unique(na.omit(y_raw)))
  if (length(levels_found) != 2) {
    stop("Column 1 ('", names(df)[1], "') must have exactly 2 distinct values; found: ",
         paste(levels_found, collapse = ", "))
  }
  if (is.null(positive_class)) {
    auto <- intersect(levels_found, c("Yes", "yes", "YES", "1", "TRUE", "True", "Case", "case", "Y"))
    if (length(auto) != 1) {
      stop("Can't tell which outcome value is the positive class (values: ",
           paste(levels_found, collapse = " / "), "). Set config[['positive_class']].")
    }
    positive_class <- auto
  }
  if (!as.character(positive_class) %in% levels_found) {
    stop("positive_class '", positive_class, "' not found in column 1.")
  }
  
  feats <- df[-1]
  non_num <- names(feats)[!vapply(feats, is.numeric, logical(1))]
  if (length(non_num) > 0) {
    stop("All feature columns must be numeric. Non-numeric: ", paste(non_num, collapse = ", "),
         "\n(Remove ID columns, or dummy-code categorical variables first.)")
  }
  if (anyNA(feats) || anyNA(y_raw)) {
    stop("Missing values found. Impute or remove them before running the pipeline.")
  }
  zero_var <- names(feats)[vapply(feats, function(v) length(unique(v)) < 2, logical(1))]
  if (length(zero_var) > 0) {
    warning("Dropping constant feature(s): ", paste(zero_var, collapse = ", "))
    feats <- feats[setdiff(names(feats), zero_var)]
  }
  
  original <- names(feats)
  safe     <- make.names(original, unique = TRUE)
  names(feats) <- safe
  
  outcome <- factor(ifelse(y_raw == as.character(positive_class), "Case", "Control"),
                    levels = c("Case", "Control"))
  
  message("Outcome '", names(df)[1], "': '", positive_class, "' = Case (n = ",
          sum(outcome == "Case"), "), other = Control (n = ", sum(outcome == "Control"),
          "); ", length(safe), " features.")
  
  list(data = data.frame(Outcome = outcome, feats, check.names = FALSE),
       labels = setNames(original, safe))
}

# Metrics
prob_case <- function(fit, newdata) predict(fit, newdata, type = "prob")[["Case"]]

pr_auc <- function(obs, p) {
  PRROC::pr.curve(scores.class0 = p[obs == "Case"],
                  scores.class1 = p[obs == "Control"])[["auc.davis.goadrich"]]
}

roc_auc <- function(obs, p) {
  as.numeric(pROC::auc(pROC::roc(obs, p, levels = c("Control", "Case"),
                                 direction = "<", quiet = TRUE)))
}

# ROC (twoClassSummary) + PR-AUC (prSummary, reported as "AUC"); "Case" = positive
pr_roc_summary <- function(data, lev = NULL, model = NULL) {
  c(caret::twoClassSummary(data, lev, model), caret::prSummary(data, lev, model))
}

make_ctrl <- function(folds, allow_parallel = TRUE, selection = "best") {
  caret::trainControl(method = "cv", number = length(folds), index = folds,
                      classProbs = TRUE, summaryFunction = pr_roc_summary,
                      savePredictions = "final", returnResamp = "final",
                      selectionFunction = selection, allowParallel = allow_parallel)
}

# Model fitting
# Fixed hyperparameters used during importance + selection (p = number of features)
fixed_grid <- function(method, p) {
  switch(method,
         svmLinear2     = data.frame(cost = 1),
         svmRadialSigma = data.frame(sigma = 1 / p, C = 1),
         glmnet         = data.frame(alpha = 0.5, lambda = 0.01),
         rf             = data.frame(mtry = max(1, floor(sqrt(p)))),
         stop("No fixed grid defined for method '", method, "'"))
}

# Full tuning grid for the final model
tuning_grid <- function(method, p) {
  switch(method,
         svmLinear2     = expand.grid(cost = 10^seq(-3, 1, length.out = 9)),
         svmRadialSigma = expand.grid(sigma = c(0.25, 0.5, 1, 2, 4) / p,
                                      C     = 10^seq(-2, 2, length.out = 9)),
         glmnet         = expand.grid(alpha  = seq(0, 1, by = 0.1),
                                      lambda = 10^seq(-4, 0, length.out = 21)),
         rf             = expand.grid(mtry = seq_len(p)),
         stop("No tuning grid defined for method '", method, "'"))
}

# One fit of `method` on `dat` (Outcome + features) with the given grid/control.
# A single-feature elastic net falls back to plain logistic regression.
fit_model <- function(method, dat, ctrl, grid = fixed_grid(method, ncol(dat) - 1), metric = "AUC") {
  p <- ncol(dat) - 1
  if (method == "glmnet" && p == 1) {
    caret::train(Outcome ~ ., data = dat, method = "glm", family = "binomial",
                 trControl = ctrl, metric = metric, preProcess = c("center", "scale"))
  } else if (method == "rf") {
    caret::train(Outcome ~ ., data = dat, method = "rf", trControl = ctrl, metric = metric,
                 tuneGrid = grid, ntree = 500)
  } else {
    caret::train(Outcome ~ ., data = dat, method = method, trControl = ctrl, metric = metric,
                 preProcess = c("center", "scale"), tuneGrid = grid)
  }
}

# Single fit with no resampling, used inside each permutation-importance fold.
# All-feature elastic net picks lambda by internal CV on that fold's training data only.
fit_fixed <- function(method, dat) {
  p <- ncol(dat) - 1
  ctrl_none <- caret::trainControl(method = "none", classProbs = TRUE)
  grid <- fixed_grid(method, p)
  if (method == "glmnet" && p > 1) {
    cvfit <- glmnet::cv.glmnet(x = as.matrix(dat[-1]), y = dat[["Outcome"]], family = "binomial",
                               alpha = 0.5, nfolds = 5, type.measure = "auc")
    grid <- data.frame(alpha = 0.5, lambda = cvfit[["lambda.1se"]])
  }
  fit_model(method, dat, ctrl_none, grid)
}

# Permutation importance
# In each fold of repeated stratified k-fold CV: fit on all features, shuffle each
# feature in the held-out fold, record the drop in PR-AUC (and ROC-AUC).
# Features are ranked by 10% trimmed mean PR-AUC drop across all folds.
perm_importance <- function(method, label, train_data, strata_train, cv_k, perm_repeats,
                            n_perm, max_features) {
  features <- setdiff(names(train_data), "Outcome")
  
  set.seed(3003)
  subsamples <- caret::createMultiFolds(droplevels(strata_train), k = cv_k, times = perm_repeats)
  n_fits <- length(subsamples)
  chunks <- split(seq_len(n_fits), ceiling(seq_len(n_fits) / 10))
  results <- vector("list", length(chunks))
  
  for (ch in seq_along(chunks)) {
    results[[ch]] <- foreach(
      b = chunks[[ch]], .combine = rbind,
      .packages = c("caret", "kernlab", "e1071", "randomForest", "glmnet", "PRROC", "pROC"),
      .export = c("fit_fixed", "fit_model", "fixed_grid", "prob_case", "pr_auc", "roc_auc")
    ) %dopar% {
      tr <- train_data[subsamples[[b]], , drop = FALSE]
      va <- train_data[-subsamples[[b]], , drop = FALSE]
      obs <- va[["Outcome"]]
      
      set.seed(1001 + b)
      fit <- fit_fixed(method, tr)
      base_p   <- prob_case(fit, va)
      base_pr  <- pr_auc(obs, base_p)
      base_roc <- roc_auc(obs, base_p)
      
      res <- do.call(rbind, lapply(features, function(f) {
        drops <- t(replicate(n_perm, {
          va_perm <- va
          va_perm[[f]] <- sample(va_perm[[f]])
          p <- prob_case(fit, va_perm)
          c(PR = base_pr - pr_auc(obs, p), ROC = base_roc - roc_auc(obs, p))
        }))
        data.frame(Subsample = b, Feature = f,
                   PR_Drop = mean(drops[, "PR"]), ROC_Drop = mean(drops[, "ROC"]))
      }))
      res[["Rank_b"]] <- rank(-(res[["PR_Drop"]] + 1e-6 * res[["ROC_Drop"]]), ties.method = "average")
      res
    }
    notify(paste0("[", label, "] permutation folds ", max(chunks[[ch]]), "/", n_fits, " done"))
  }
  
  dplyr::bind_rows(results) %>%
    dplyr::group_by(Feature) %>%
    dplyr::summarise(Mean_Rank  = mean(Rank_b),
                     Top_Freq   = mean(Rank_b <= max_features),
                     PR_Drop_SD = sd(PR_Drop),
                     PR_Drop    = mean(PR_Drop, trim = 0.1),
                     ROC_Drop   = mean(ROC_Drop, trim = 0.1),
                     .groups    = "drop") %>%
    dplyr::arrange(dplyr::desc(PR_Drop), dplyr::desc(ROC_Drop)) %>%
    dplyr::mutate(Rank = dplyr::row_number())
}

# Forward feature selection along a ranking
# Adds features one at a time in importance order; records CV PR-AUC (+95% CI) and ROC-AUC.
forward_select <- function(method, label, ranked, train_data, ctrl, max_features) {
  n_steps <- min(max_features, length(ranked))
  log <- vector("list", n_steps)
  
  for (i in seq_len(n_steps)) {
    feats <- ranked[seq_len(i)]
    dat   <- train_data[c("Outcome", feats)]
    fit   <- fit_model(method, dat, ctrl)
    
    resamp <- fit[["resample"]]
    pr     <- resamp[["AUC"]]
    n_ok   <- sum(!is.na(pr))
    m_pr   <- mean(pr, na.rm = TRUE)
    half   <- 1.96 * sd(pr, na.rm = TRUE) / sqrt(n_ok)
    
    log[[i]] <- data.frame(Size = i, Feature = ranked[i], PR_AUC = m_pr,
                           Lower_CI = m_pr - half, Upper_CI = m_pr + half,
                           ROC_AUC = mean(resamp[["ROC"]], na.rm = TRUE))
    notify(sprintf("[%s] step %d/%d, added %s | PR-AUC %.3f", label, i, n_steps, ranked[i], m_pr))
  }
  dplyr::bind_rows(log)
}

##### Data Preprocessing #####
dir.create(config[["out_dir"]], showWarnings = FALSE, recursive = TRUE)

raw_data <- if (exists("input_data")) input_data else read_input(config[["data_file"]])
prepped  <- prepare_data(raw_data, config[["positive_class"]])
ML_Data  <- prepped[["data"]]
labels   <- prepped[["labels"]]
label_of <- function(x) unname(ifelse(x %in% names(labels), labels[x], x))

strata <- if (is.null(config[["strata"]])) ML_Data[["Outcome"]] else factor(config[["strata"]])
stopifnot(length(strata) == nrow(ML_Data), !anyNA(strata))

set.seed(config[["seed"]])
train_idx <- caret::createDataPartition(strata, p = config[["train_prop"]], list = FALSE)[, 1]
trainData <- ML_Data[train_idx, , drop = FALSE]
testData  <- ML_Data[-train_idx, , drop = FALSE]
strata_train <- strata[train_idx]

print(table(Set = rep(c("Train", "Test"), c(nrow(trainData), nrow(testData))),
            Outcome = c(as.character(trainData[["Outcome"]]), as.character(testData[["Outcome"]]))))

set.seed(config[["seed"]])
cv_folds <- caret::createFolds(strata_train, k = config[["cv_k"]], returnTrain = TRUE)

prevalence   <- mean(trainData[["Outcome"]] == "Case")
n_features   <- ncol(trainData) - 1
max_features <- if (is.null(config[["max_features"]])) floor(nrow(trainData) / 10) else config[["max_features"]]
max_features <- max(1, min(max_features, n_features))

models      <- config[["models"]]
model_cols  <- setNames(c("#de8f05", "#0173b2", "#029e73", "#cc78bc", "#ca9161", "#949494")[seq_along(models)],
                        names(models))
imp_name    <- function(m) paste(m, "Permutation Importance")
ffs_name    <- function(m) paste(m, "Ranked FFS PRAUC")

##### Permutation Importance and Forward Feature Selection #####
importance <- list()
selection  <- list()

if (config[["run_selection"]]) {
  cl <- parallel::makeCluster(config[["n_cores"]],
                              outfile = file.path(config[["out_dir"]], "parallel_log.txt"))
  doParallel::registerDoParallel(cl)
  on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)
  notify(paste0("Parallel processing started on ", config[["n_cores"]], " cores."))
  
  for (m in names(models)) {
    method <- models[[m]]
    notify(paste0("Starting ", m))
    
    importance[[m]] <- perm_importance(method, m, trainData, strata_train,
                                       cv_k = config[["cv_k"]], perm_repeats = config[["perm_repeats"]],
                                       n_perm = config[["n_perm"]], max_features = max_features)
    writexl::write_xlsx(importance[[m]], out_file(imp_name(m)))
    
    set.seed(config[["seed"]])
    selection[[m]] <- forward_select(method, m, importance[[m]][["Feature"]], trainData,
                                     ctrl = make_ctrl(cv_folds, allow_parallel = (method != "rf")),
                                     max_features = max_features)
    writexl::write_xlsx(selection[[m]], out_file(ffs_name(m)))
    notify(paste0(m, " complete."))
  }
  
  parallel::stopCluster(cl)
  foreach::registerDoSEQ()
} else {
  for (m in names(models)) {
    importance[[m]] <- as.data.frame(readxl::read_xlsx(out_file(imp_name(m))))
    selection[[m]]  <- as.data.frame(readxl::read_xlsx(out_file(ffs_name(m))))
  }
}

##### Model Selection #####
# Permutation importance plot
ImpDf <- dplyr::bind_rows(lapply(names(importance), function(m) {
  importance[[m]] %>% dplyr::slice_head(n = 15) %>% dplyr::mutate(Model = m)
})) %>%
  dplyr::mutate(Label = paste0(label_of(Feature), "___", Model),
                Label = reorder(factor(Label), PR_Drop))

ImpPlot <- ggplot(ImpDf, aes(x = PR_Drop, y = Label, fill = Model)) +
  geom_col() +
  geom_errorbar(aes(xmin = PR_Drop - PR_Drop_SD, xmax = PR_Drop + PR_Drop_SD),
                width = 0.3, colour = "grey30", orientation = "y") +
  scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
  facet_wrap(~ Model, scales = "free_y") +
  scale_fill_manual(values = model_cols) +
  labs(x = "Mean drop in PR-AUC when shuffled", y = NULL,
       title = paste0("Permutation Importance: Top 15 Features (", config[["run_name"]], ")"),
       subtitle = paste0("10% trimmed mean PR-AUC drop across ", config[["cv_k"]], "-fold stratified CV x ",
                         config[["perm_repeats"]], " repeats (", config[["n_perm"]], " shuffles per fold); bars +/- SD")) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")
save_plot(ImpPlot, "Permutation Importance", width = 4500, height = 3500)

# Forward selection curves and choice of best model / panel size
AllSel <- dplyr::bind_rows(lapply(names(selection), function(m) {
  selection[[m]] %>% dplyr::mutate(Model = m, Label = toupper(label_of(Feature)))
}))

PR_FFS <- ggplot(AllSel, aes(x = Size, y = PR_AUC, colour = Model)) +
  geom_hline(yintercept = prevalence, linetype = "dashed", colour = "grey50") +
  geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI, fill = Model), colour = NA, alpha = 0.1) +
  geom_point(size = 1.5, na.rm = TRUE) +
  geom_line() +
  geom_text_repel(aes(label = Label), colour = "black", size = 3, segment.color = NA, show.legend = FALSE) +
  scale_colour_manual(values = model_cols) +
  scale_fill_manual(values = model_cols, guide = "none") +
  labs(x = "Number of features", y = paste0("PR-AUC (", config[["cv_k"]], "-fold CV)"),
       title = paste0("Permutation-Ranked Forward Feature Selection (", config[["run_name"]], ")"),
       subtitle = paste0("95% CIs from ", config[["cv_k"]], "-fold stratified CV; dashed line = event rate ",
                         round(prevalence, 2))) +
  theme_bw(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank())
save_plot(PR_FFS, "All Comparisons Ranked FFS PRAUC", width = 4000, height = 2500)

model_summary <- AllSel %>%
  dplyr::group_by(Model) %>%
  dplyr::mutate(SE      = (Upper_CI - Lower_CI) / (2 * 1.96),
                Peak    = max(PR_AUC),
                Peak_SE = SE[which.max(PR_AUC)]) %>%
  dplyr::ungroup()

best_row         <- model_summary %>% dplyr::arrange(dplyr::desc(PR_AUC), Size) %>% dplyr::slice(1)
final_model_name <- best_row[["Model"]]
final_method     <- models[[final_model_name]]

final_n_feats <- if (config[["use_one_se"]]) {
  model_summary %>%
    dplyr::filter(Model == final_model_name, PR_AUC >= Peak - Peak_SE) %>%
    dplyr::summarise(n = min(Size)) %>%
    dplyr::pull(n)
} else {
  best_row[["Size"]]
}

print(model_summary %>% dplyr::group_by(Model) %>% dplyr::slice_max(PR_AUC, n = 1, with_ties = FALSE))
cat("Selected model:", final_model_name, "| Panel size:", final_n_feats,
    "| Peak CV PR-AUC:", round(best_row[["PR_AUC"]], 3), "at", best_row[["Size"]], "features\n")
writexl::write_xlsx(model_summary, out_file("Ranked FFS Model Summary PRAUC"))

# Held-out test of every panel (reported only, not used to choose the model)
panel_test <- dplyr::bind_rows(lapply(names(selection), function(m) {
  sel    <- selection[[m]] %>% dplyr::arrange(Size)
  method <- models[[m]]
  dplyr::bind_rows(lapply(seq_len(nrow(sel)), function(k) {
    feats <- sel[["Feature"]][seq_len(k)]
    set.seed(config[["seed"]])
    fit <- fit_model(method, trainData[c("Outcome", feats)], make_ctrl(cv_folds, FALSE))
    p   <- prob_case(fit, testData[c("Outcome", feats)])
    obs <- testData[["Outcome"]]
    data.frame(Model = m, Size = k, Feature_added = label_of(sel[["Feature"]][k]),
               CV_PR_AUC = sel[["PR_AUC"]][k], Test_PR_AUC = pr_auc(obs, p),
               CV_ROC_AUC = sel[["ROC_AUC"]][k], Test_ROC_AUC = roc_auc(obs, p))
  }))
}))
print(panel_test)
writexl::write_xlsx(panel_test, out_file("Ranked FFS Panel Test PRAUC"))

PanelPlot <- panel_test %>%
  tidyr::pivot_longer(c(CV_PR_AUC, Test_PR_AUC), names_to = "Estimate", values_to = "PR_AUC") %>%
  dplyr::mutate(Estimate = ifelse(Estimate == "CV_PR_AUC", "Cross-validated (selection)", "Held-out test")) %>%
  ggplot(aes(x = Size, y = PR_AUC, colour = Model, linetype = Estimate)) +
  geom_hline(yintercept = prevalence, colour = "grey70", linetype = "dashed") +
  geom_line() +
  geom_point(size = 1.5) +
  scale_colour_manual(values = model_cols) +
  scale_x_continuous(breaks = seq_len(max_features)) +
  labs(x = "Number of features", y = "PR-AUC",
       title = paste0("Forward-Selected Panels: CV vs Held-out Test (", config[["run_name"]], ")")) +
  theme_bw(base_size = 14) +
  theme(legend.position = "top", legend.box = "vertical", legend.title = element_blank())
save_plot(PanelPlot, "Ranked FFS CV vs Test PRAUC", width = 4000, height = 2500)

##### Final Model #####
SelectionResult <- selection[[final_model_name]] %>% dplyr::arrange(Size)
OptimalFeats    <- SelectionResult[["Feature"]][seq_len(final_n_feats)]
stopifnot(all(OptimalFeats %in% names(trainData)))
cat("Optimal features:", paste(label_of(OptimalFeats), collapse = ", "), "\n")

trainSub <- trainData[c("Outcome", OptimalFeats)]
testSub  <- testData[c("Outcome", OptimalFeats)]
p_final  <- length(OptimalFeats)

set.seed(config[["seed"]])
final_model <- fit_model(final_method, trainSub,
                         ctrl = make_ctrl(cv_folds, allow_parallel = FALSE, selection = "oneSE"),
                         grid = tuning_grid(final_method, p_final))

print(final_model[["bestTune"]])
cv_pr_final  <- mean(final_model[["resample"]][["AUC"]], na.rm = TRUE)
cv_roc_final <- mean(final_model[["resample"]][["ROC"]], na.rm = TRUE)
print(c(Tuned_CV_PR_AUC = cv_pr_final, Tuned_CV_ROC_AUC = cv_roc_final,
        Selection_PR_AUC = SelectionResult[["PR_AUC"]][final_n_feats]))

saveRDS(final_model, out_file("Optimised Final Model", ".rds"))

# Youden threshold from training out-of-fold predictions
oof_train <- final_model[["pred"]] %>%
  dplyr::group_by(rowIndex) %>%
  dplyr::summarise(obs = dplyr::first(obs), Case = mean(Case), .groups = "drop")

roc_train  <- pROC::roc(oof_train[["obs"]], oof_train[["Case"]], levels = c("Control", "Case"),
                        direction = "<", quiet = TRUE)
youden_thr <- pROC::coords(roc_train, x = "best", best.method = "youden",
                           ret = c("threshold", "sensitivity", "specificity"), transpose = FALSE)
print(youden_thr)
thr <- youden_thr[["threshold"]][1]

classify <- function(fit, newdata) {
  factor(ifelse(prob_case(fit, newdata) >= thr, "Case", "Control"), levels = c("Case", "Control"))
}

##### Decision Boundary #####
Xf <- trainSub[-1]
yf <- trainSub[["Outcome"]]

# Maps a grid of PC scores (first n_dims PCs, rest = 0) back to the original feature space
pcs_to_features <- function(pc_grid, pca, n_dims) {
  full <- matrix(0, nrow = nrow(pc_grid), ncol = ncol(pca[["x"]]))
  full[, seq_len(n_dims)] <- as.matrix(pc_grid[seq_len(n_dims)])
  orig <- full %*% t(pca[["rotation"]])
  orig <- sweep(orig, 2, pca[["scale"]], `*`)
  orig <- sweep(orig, 2, pca[["center"]], `+`)
  setNames(as.data.frame(orig), colnames(Xf))
}

make_pc_grid <- function(scores, n_dims, resolution) {
  axes <- lapply(seq_len(n_dims), function(d) {
    r <- range(scores[, d]); m <- 0.1 * diff(r)
    seq(r[1] - m, r[2] + m, length.out = resolution)
  })
  setNames(expand.grid(axes), paste0("PC", seq_len(n_dims)))
}

if (ncol(Xf) >= 2) {
  pca_result <- prcomp(Xf, center = TRUE, scale. = TRUE)
  var_expl   <- summary(pca_result)[["importance"]][2, ]
  pca_scores <- pca_result[["x"]]
  
  pca_data <- data.frame(PC1 = pca_scores[, 1], PC2 = pca_scores[, 2], class = yf)
  grid2d   <- make_pc_grid(pca_scores, 2, 300)
  grid2d[["Prediction"]] <- classify(final_model, pcs_to_features(grid2d, pca_result, 2))
  
  p_boundary <- ggplot() +
    geom_tile(data = grid2d, aes(x = PC1, y = PC2, fill = Prediction), alpha = 0.3) +
    geom_point(data = pca_data, aes(x = PC1, y = PC2, colour = class), size = 3) +
    scale_fill_manual(values = c("Control" = "skyblue", "Case" = "salmon"), name = "Prediction") +
    scale_colour_manual(values = c("Control" = "blue", "Case" = "red"), name = "Actual") +
    labs(title = paste0("Final ", final_model_name, " (", final_n_feats, " features): Decision Boundary in PCA Space"),
         subtitle = paste0("CV PR-AUC: ", round(cv_pr_final, 3), " | Youden threshold: ", round(thr, 3)),
         x = paste0("PC1 (", round(var_expl[1] * 100, 1), "% variance)"),
         y = paste0("PC2 (", round(var_expl[2] * 100, 1), "% variance)")) +
    theme_bw(base_size = 16) +
    theme(legend.position = "top")
  save_plot(p_boundary, "Decision Boundary")
  
  write.csv(pca_data, out_file("training_pca_data", ".csv"), row.names = FALSE)
  write.csv(grid2d,   out_file("grid_2d", ".csv"), row.names = FALSE)
  
  if (ncol(Xf) >= 3) {
    grid3d <- make_pc_grid(pca_scores, 3, 50)
    grid3d[["Prediction"]] <- classify(final_model, pcs_to_features(grid3d, pca_result, 3))
    write.csv(grid3d, out_file("grid_3d", ".csv"), row.names = FALSE)
    write.csv(data.frame(pca_scores[, 1:3], class = yf), out_file("training_pca_data_3d", ".csv"),
              row.names = FALSE)
  }
}

##### SHAP #####
if (config[["run_shap"]]) {
  shap_rows <- seq_len(nrow(trainSub))
  if (!is.null(config[["shap_max_obs"]]) && config[["shap_max_obs"]] < length(shap_rows)) {
    set.seed(config[["seed"]])
    shap_rows <- sort(sample(shap_rows, config[["shap_max_obs"]]))
  }
  
  # iml uses R6 classes: `$new()` is a constructor call, not column access
  predictor <- iml::Predictor$new(final_model, data = Xf, type = "prob")
  
  shap_list <- lapply(seq_along(shap_rows), function(j) {
    i <- shap_rows[j]
    res <- iml::Shapley$new(predictor, x.interest = Xf[i, , drop = FALSE])[["results"]]
    res <- res[res[["class"]] == "Case", c("feature", "phi")]
    if (j %% 25 == 0) message("SHAP ", j, "/", length(shap_rows))
    data.frame(obs_id = i, Feature = res[["feature"]], SHAP_Value = res[["phi"]])
  })
  SHAP_long <- dplyr::bind_rows(shap_list)
  
  FeatureVals_long <- as.data.frame(scale(Xf)) %>%
    dplyr::mutate(obs_id = seq_len(nrow(Xf))) %>%
    tidyr::pivot_longer(-obs_id, names_to = "Feature", values_to = "Feature_Value")
  
  combined_long <- dplyr::inner_join(SHAP_long, FeatureVals_long, by = c("obs_id", "Feature")) %>%
    dplyr::mutate(Label = label_of(Feature))
  
  feat_order <- combined_long %>%
    dplyr::group_by(Label) %>%
    dplyr::summarise(m = mean(abs(SHAP_Value)), .groups = "drop") %>%
    dplyr::arrange(m) %>%
    dplyr::pull(Label)
  
  SHAP_plot <- ggplot(combined_long, aes(x = SHAP_Value, y = Label, colour = Feature_Value)) +
    geom_jitter(width = 0.01, height = 0.2) +
    scale_colour_gradient2(low = "dodgerblue3", mid = "gray90", high = "firebrick3", midpoint = 0,
                           name = "Feature value (scaled)") +
    scale_y_discrete(limits = feat_order) +
    labs(title = "SHAP Summary Plot",
         subtitle = paste0("Shapley values for ", length(shap_rows), " training observations"),
         x = "SHAP value (effect on P(Case))", y = NULL) +
    theme_bw(base_size = 16) +
    theme(legend.position = "top", legend.key.width = unit(1.5, "cm"))
  save_plot(SHAP_plot, "SHapley Additive exPlanations")
}

##### Test Set Evaluation #####
obs_test  <- testSub[["Outcome"]]
p_test    <- prob_case(final_model, testSub)
pred_test <- factor(ifelse(p_test >= thr, "Case", "Control"), levels = c("Case", "Control"))
cm        <- caret::confusionMatrix(pred_test, obs_test, positive = "Case")
cm_stats  <- cm[["byClass"]]
print(cm)

# Confusion matrix
cm_df <- as.data.frame(table(Predicted = pred_test, Actual = obs_test))
cm_df[["Predicted"]] <- factor(cm_df[["Predicted"]], levels = c("Case", "Control"))
cm_df[["Actual"]]    <- factor(cm_df[["Actual"]],    levels = c("Control", "Case"))

CM_plot <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Freq), size = 8) +
  scale_fill_gradientn(colours = rev(hcl.colors(25, "OrRd"))) +
  labs(title = "Confusion Matrix: Youden Threshold",
       subtitle = paste0("Threshold = ", round(thr, 3), " (from training out-of-fold predictions)"),
       x = "Actual outcome", y = "Predicted outcome") +
  theme_bw(base_size = 18) +
  theme(legend.position = "top", legend.key.width = unit(1.5, "cm"))
save_plot(CM_plot, "Confusion Matrix Youden")

# ROC with bootstrap CI
roc_test <- pROC::roc(obs_test, p_test, levels = c("Control", "Case"), direction = "<",
                      ci = TRUE, quiet = TRUE)
set.seed(config[["seed"]])
se_ci  <- pROC::ci.se(roc_test, conf.level = 0.95, method = "bootstrap", boot.n = 5000,
                      boot.stratified = TRUE, specificities = seq(0, 1, l = 100), plot = FALSE)
roc_ci_df <- data.frame(spec = as.numeric(rownames(se_ci)), lower = se_ci[, 1], upper = se_ci[, 3])
auc_ci <- pROC::ci.auc(roc_test, conf.level = 0.95, method = "bootstrap", boot.n = 5000)
test_roc_auc <- as.numeric(pROC::auc(roc_test))

ROC_plot <- pROC::ggroc(roc_test, legacy.axes = TRUE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey") +
  geom_ribbon(data = roc_ci_df, aes(x = 1 - spec, ymin = lower, ymax = upper),
              fill = "steelblue", alpha = 0.2, inherit.aes = FALSE) +
  annotate("point", x = 1 - cm_stats[["Specificity"]], y = cm_stats[["Sensitivity"]],
           colour = "firebrick3", size = 3) +
  annotate("label", x = 0.65, y = 0.2, size = 5,
           label = sprintf("AUC = %.3f\n95%% CI [%.3f-%.3f]", test_roc_auc, auc_ci[1], auc_ci[3])) +
  coord_equal() +
  labs(x = "1 - Specificity", y = "Sensitivity",
       title = paste0(final_model_name, " ROC Curve, Test Set (", config[["run_name"]], ")"),
       subtitle = "Red point = Youden threshold") +
  theme_bw(base_size = 16)
save_plot(ROC_plot, "Optimal Model ROC Curve", width = 2000, height = 2000)

# PR curve with stratified bootstrap CI
pr_test <- PRROC::pr.curve(scores.class0 = p_test[obs_test == "Case"],
                           scores.class1 = p_test[obs_test == "Control"], curve = TRUE)
test_pr_auc <- pr_test[["auc.davis.goadrich"]]

set.seed(config[["seed"]])
case_idx <- which(obs_test == "Case")
ctrl_idx <- which(obs_test == "Control")
boot_pr  <- replicate(5000, {
  idx <- c(sample(case_idx, replace = TRUE), sample(ctrl_idx, replace = TRUE))
  pr_auc(obs_test[idx], p_test[idx])
})
pr_ci     <- quantile(boot_pr, c(0.025, 0.975), na.rm = TRUE)
test_prev <- mean(obs_test == "Case")
pr_curve  <- pr_test[["curve"]]

PR_plot <- ggplot(data.frame(Recall = pr_curve[, 1], Precision = pr_curve[, 2]),
                  aes(x = Recall, y = Precision)) +
  geom_path(linewidth = 1) +
  geom_hline(yintercept = test_prev, linetype = "dashed", colour = "grey") +
  annotate("point", x = cm_stats[["Sensitivity"]], y = cm_stats[["Pos Pred Value"]],
           colour = "firebrick3", size = 3) +
  annotate("label", x = 0.65, y = 0.85, size = 5,
           label = sprintf("PR-AUC = %.3f\n95%% CI [%.3f-%.3f]\nBaseline = %.3f",
                           test_pr_auc, pr_ci[1], pr_ci[2], test_prev)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = paste0(final_model_name, " PR Curve, Test Set (", config[["run_name"]], ")"),
       subtitle = "Red point = Youden threshold; dashed = event rate") +
  theme_bw(base_size = 16)
save_plot(PR_plot, "Optimal Model PR Curve", width = 2000, height = 2000)

# Final metrics
FinalMetrics <- data.frame(
  Model             = final_model_name,
  Features          = paste(label_of(OptimalFeats), collapse = ", "),
  Best_Tune         = paste(names(final_model[["bestTune"]]), signif(unlist(final_model[["bestTune"]]), 3),
                            sep = "=", collapse = "; "),
  CV_PR_AUC         = cv_pr_final,
  CV_ROC_AUC        = cv_roc_final,
  Youden_Threshold  = thr,
  Test_PR_AUC       = test_pr_auc,
  Test_PR_AUC_Lower = pr_ci[[1]],
  Test_PR_AUC_Upper = pr_ci[[2]],
  Test_ROC_AUC      = test_roc_auc,
  Test_ROC_Lower    = auc_ci[1],
  Test_ROC_Upper    = auc_ci[3],
  Sensitivity       = cm_stats[["Sensitivity"]],
  Specificity       = cm_stats[["Specificity"]],
  PPV               = cm_stats[["Pos Pred Value"]],
  NPV               = cm_stats[["Neg Pred Value"]],
  Balanced_Accuracy = cm_stats[["Balanced Accuracy"]],
  Brier             = mean((p_test - as.numeric(obs_test == "Case"))^2)
)
print(t(FinalMetrics))
writexl::write_xlsx(FinalMetrics, out_file("Final Model Test Metrics"))

notify(sprintf("All done! %s (%d features) | Test PR-AUC %.3f | ROC-AUC %.3f",
               final_model_name, final_n_feats, test_pr_auc, test_roc_auc))