---
title: "Statilytics Studio: a desktop toolkit for applied statistics and reproducible reporting"
tags:
  - statistics
  - data analysis
  - GUI
  - PySide6
  - statsmodels
  - scikit-learn
  - epidemiology
  - clinical research
authors:
  - name: Mirza Niaz Zaman Elin
    affiliation: 1
affiliations:
  - name: AMAL Youth & Family Centre, St. John’s, NL, Canada
    index: 1
date: 2025-09-03
bibliography: paper.bib
---

# Summary
Statilytics Studio is a Python-based desktop application that integrates data ingestion, exploratory analysis,and statistical inference within a single GUI. The application emphasizes pragmatic workflows common in public health, social science, education, and clinical research. Results are rendered into shareable HTML/DOCX reports to encourage transparent and reproducible analysis (via captured tables, figures, and metadata). The analytical core is importable as a standalone module for scripted use. This application is a coding-free and user-friendly way to conduct statistical analysis without any extraordinary technical knowledge.

# Statement of need
Many researchers in health and social sciences rely on a mixture of spreadsheet tools and black‑box statistical packages. Open, scriptable environments lower barriers to transparency but can pose usability hurdles for non‑programmers. Statilytics Studio addresses this gap by combining an approachable PySide6 interface with the open scientific Python stack. Typical user stories include: comparing groups with parametric/non‑parametric tests; running regression families (including mixed models and GLMs); computing reliability/validity metrics; producing survival curves and diagnostic ROC/PR analyses; clustering and PCA/EFA for structure discovery; and exporting publication‑ready tables and figures. Moreover, the software ensures ease of use without any coding or specialized expertise with statistical software. The automated process that includes data uploading and automated analysis, saves researchers' valuable time as well.

# State of the field
The project builds on mature, peer‑reviewed libraries: NumPy, pandas, SciPy, statsmodels, and scikit‑learn [@harris2020; @mckinney2010; @virtanen2020; @seabold2010; @pedregosa2011]. Optional capabilities rely on widely used packages for survival analysis, reliability, factor/correspondence analysis, and document generation [@DavidsonPilon2019; @vallat2018; @factor_analyzer; @prince; @krippendorff; @python_docx]. Desktop statistical front‑ends such as jamovi and JASP demonstrate the value of GUI‑first workflows layered atop open methods [@jamovi; @JASP2025]. Statilytics Studio contributes a single‑file Qt application oriented toward lightweight, local analysis with emphasis on reproducible reporting.

# Functionality
The GUI supports: descriptive statistics and crosstabs; parametric and non‑parametric tests; correlations; OLS/logit/multinomial/ordinal/GLM; linear mixed models; reliability and agreement metrics; survival analysis (KM, Cox PH, Weibull AFT); ROC/PR and Brier score; ARIMA and ETS forecasting; random‑effects (DL) meta‑analysis; PCA/EFA/CA; LDA/trees/random forests; K‑means/agglomerative and auto‑KMeans. The `statilytics_studio.core.Engine` exposes these methods for scripted workflows, enabling unit testing without a GUI dependency.

# Conflict of interest
The author declares that there are no competing interests—financial or non-financial—related to this work.

# Acknowledgements
Gratitude is extended to contributors and to the maintainers of the scientific Python ecosystem.

# References
