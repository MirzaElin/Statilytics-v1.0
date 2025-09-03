# Statilytics Studio

A desktop application for applied statistics and machine learning with reproducible reporting. The interface is built with PySide6 and integrates `pandas`, `NumPy`, `SciPy`, `statsmodels`, and `scikit-learn`, with optional modules for survival analysis, psychometrics, factor analysis, correspondence analysis, and document export.

## Features (non-exhaustive)
- Data ingestion from CSV and Excel.
- Descriptive statistics and crosstabs with inference (χ², Fisher's exact).
- Parametric and non-parametric tests (t, ANOVA, ANCOVA, MANOVA/MANCOVA, Mann–Whitney U, Wilcoxon, Kruskal–Wallis, Friedman, Tukey HSD).
- Correlations (Pearson, Spearman, Kendall).
- Regression families (OLS, logistic, multinomial logit, ordinal logit, GLM Poisson/NegBin/Gamma) and linear mixed models.
- Reliability and agreement (Cronbach’s α, Cohen’s κ, weighted κ, Scott’s π, Fleiss’ κ, ICC, Krippendorff’s α).
- Survival analysis (Kaplan–Meier, Cox PH, Weibull AFT) when `lifelines` is available.
- Diagnostic curves (ROC, PR, Brier score).
- Time‑series (ARIMA, ETS), meta‑analysis (DerSimonian–Laird).
- Dimension reduction and clustering (PCA, EFA, LDA, decision tree/CHAID‑like, random forest, K‑means, agglomerative, auto‑KMeans).
- Correspondence analysis via `prince`.
- Reproducible HTML/DOCX report export.

## Installation
Installation can be performed with pip in a fresh environment. Minimal requirements for programmatic use of the analytical engine:
```bash
pip install numpy pandas scipy statsmodels scikit-learn
```
Installation of the full desktop application with optional extras:
```bash
pip install "PySide6>=6.6" "matplotlib>=3.7" "python-docx>=1.0.0" "openpyxl>=3.1"                 "lifelines>=0.27" "pingouin>=0.5" "factor-analyzer>=0.5" "prince>=0.13" "krippendorff>=0.6"
```

## Launch
After installation as a package, the application entry point is `statilytics-studio`.
A module entry point is also provided: `python -m statilytics_studio.app`.

## Programmatic usage
The analytical engine can be imported independently of the GUI:
```python
import pandas as pd
from statilytics_studio.core import Engine

df = pd.DataFrame({"x":[1,2,3], "y":[2,3,4]})
eng = Engine(df)
desc = eng.describe()
print(desc.head())
```

## License
MIT License © 2025 Mirza Niaz Zaman Elin.

## Acknowledgements
This package builds upon the scientific Python ecosystem, notably: NumPy, pandas, SciPy, statsmodels, scikit‑learn, and several optional libraries cited in the paper.
