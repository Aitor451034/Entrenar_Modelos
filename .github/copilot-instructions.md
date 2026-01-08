# AI Coding Assistant Instructions for Welding Defect Detection Project

## Project Overview
This is a machine learning project for detecting welding fusion faults in titanium sheets using resistance spot welding data. The system analyzes 32 statistical parameters extracted from dynamic resistance curves (time-series data) to classify welds as defective (1) or acceptable (0).

## Architecture & Data Flow
- **Raw Data**: CSV files with welding parameters and embedded time-series curves (Curv_I, Curv_V)
- **Feature Engineering**: Extract 32 statistical features (ranges, deviations, slopes, areas, curvatures, etc.) from resistance curves
- **Modeling**: Train classifiers using imbalanced-learn pipelines with scaling, feature selection (RFE), and ensemble methods
- **Evaluation**: Optimize decision thresholds for F-beta score (β=4) to minimize false negatives (defective welds misclassified as good)
- **Output**: Saved pipeline models (.pkl) with optimized thresholds, evaluation reports in Markdown

## Key Conventions & Patterns
- **Imbalanced Data Handling**: Use `BalancedRandomForestClassifier` or `ImbPipeline` for class imbalance (few defects)
- **Pipelines**: Always use `ImbPipeline` with `StandardScaler` → `RFE` (RandomForest-based) → classifier
- **Hyperparameter Tuning**: `RandomizedSearchCV` with `RepeatedStratifiedKFold` (5 splits, 3 repeats)
- **Threshold Optimization**: Binary search for threshold maximizing F-beta score with minimum precision constraint (e.g., 0.70)
- **Feature Names**: Use predefined `FEATURE_NAMES` list with 32 statistical parameters
- **File Selection**: Scripts use `tkinter.filedialog` for interactive CSV loading
- **Model Saving**: Save complete pipelines (scaler + selector + model + threshold) as `.pkl` files
- **Evaluation Metrics**: Focus on precision, recall, F-beta, confusion matrix, ROC-AUC; report on test set only

## Developer Workflows
- **Data Preparation**: Run `Discriminacion_datos.py` to analyze defect distribution in raw data
- **Model Training**: Execute training scripts in `SRC/` subfolders (e.g., `RandomForest/Sin_Smote/BalancedRandomForest_sin_Smote.py`)
- **Feature Extraction**: Reuse feature engineering functions from existing scripts (signal processing with scipy)
- **Validation**: Always evaluate on held-out test set (30% stratified split); avoid overfitting to train set
- **Results Documentation**: Generate Markdown reports with confusion matrices and ROC curves

## Dependencies & Environment
- **Virtual Environment**: Use `env/` with Python 3.x
- **Key Libraries**: `imbalanced-learn`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- **ML Algorithms**: `BalancedRandomForestClassifier`, `CatBoostClassifier`, `XGBClassifier`, `LGBMClassifier`, SVM variants
- **Install**: `pip install -r requirements.txt`

## Code Examples
- **Pipeline Structure**: See `SRC/RandomForest/Sin_Smote/BalancedRandomForest_sin_Smote.py` lines 50-100
- **Feature Engineering**: Reuse statistical calculations from any training script
- **Threshold Optimization**: Implement binary search for F-beta maximization (reference existing scripts)
- **Model Persistence**: Use `pickle.dump()` for full pipeline + threshold dictionary

## Important Notes
- **Language**: Code comments in Spanish, but maintain English for library functions
- **Defect Classes**: 0=acceptable, 1=fusion fault; prioritize minimizing false negatives
- **Data Paths**: Hardcoded paths in scripts (e.g., `C:\Users\...\DATA\`) - update for your environment
- **Reproducibility**: Set `RANDOM_STATE_SEED = 42` for consistent results</content>
<parameter name="filePath">c:\Users\aitor\Desktop\Entrenar_Modelos\.github\copilot-instructions.md