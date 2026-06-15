# tffit: Python implementation of radiocesium transfer factor models for wheat

This repository provides a Python implementation of soil-to-wheat radiocesium (<sup>137</sup>Cs) transfer factor (TF) models analyzed in the following published journal article:

- Seki, K., Yamaguchi, N., Eguchi, T., Igura, M., 2026. Improvement of spatiotemporal generalization in radiocesium transfer models for wheat using symbolic regression. *Journal of Environmental Radioactivity* 298, 108077. https://doi.org/10.1016/j.jenvrad.2026.108077

It is designed to ensure transparency and reproducibility by allowing readers and reviewers to execute model fitting, external cross-validation, supplementary uncertainty analyses, and plotting using the Supplementary dataset.

## What this repository is for

1. **Reproducibility:** Reproduce the fitted coefficients, validation metrics, and supplementary uncertainty analyses reported in the paper.
2. **Reusability:** Provide a reusable codebase for soil-to-crop radiocesium transfer analyses, including alternative model formulations, validation schemes, uncertainty analyses, and visualization.

## Data availability

This repository contains **code only** and does **not** include the dataset used in the paper.

The dataset is provided as **Supplementary Material Table S1** of the [published journal article](https://doi.org/10.1016/j.jenvrad.2026.108077).

## Features

- Fits multiple radiocesium transfer factor models selectable by `--model`.
- Uses log<sub>10</sub>(TF) as the target variable.
- Reproduces fitted coefficients and in-sample metrics for Table 4.
- Performs LOSO and LOYO cross-validation for Table 5.
- Reports per-year LOYO RMSE values.
- Generates coefficient bootstrap confidence intervals for Table S3.
- Performs paired out-of-fold error comparisons for Table S4.
- Outputs cluster-level loss differences for Table S5.
- Supports observed vs. predicted scatter plots.
- Supports text, CSV, and Markdown table output.
- Provides strict input validation for log-transformed variables.

## Models

All equations use log = log<sub>10</sub>.

| Model name | Equation |
|---|---|
| absalom | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(min(K<sub>ex</sub>/CEC, k<sub>lim</sub>)) - log(RIP) |
| k       | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) |
| kr      | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) - k<sub>3</sub> log(RIP) |
| krc     | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) + k<sub>3</sub> log(RIP) + k<sub>4</sub> log(CEC) |
| krp     | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) - k<sub>3</sub> log(RIP) - k<sub>4</sub> pH |
| krcs    | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) - k<sub>3</sub> log(RIP) + k<sub>4</sub> log(Cs) |
| sr1     | log(TF) = -(RIP - log(RIP)) - k<sub>1</sub>(k<sub>2</sub> - RIP)K<sub>ex</sub> |
| sr2     | log(TF) = -0.85 RIP - k<sub>1</sub> max(k<sub>2</sub> - RIP, 0)K<sub>ex</sub> |

**Variables and units, as in Supplementary Table S1:**

- `TF`: transfer factor, dimensionless
- `Ex-K`: exchangeable potassium, mol/kg
- `RIP`: radiocesium interception potential, mol/kg
- `CEC`: cation exchange capacity, molc/kg
- `137Cs`: radiocesium concentration in soil, Bq/kg
- `pH`: soil pH, water, 1:5

Notes:

- K<sub>ex</sub> corresponds to the `Ex-K` column.
- Cs in the model equations corresponds to the `137Cs` column.

## Installation

### Windows users

Use [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl/) (WSL) to run this project.

### Setup on Linux, macOS, or WSL

1. Install git if it is not already installed.

```bash
# WSL / Ubuntu / Debian
sudo apt update
sudo apt install git -y

# macOS with Homebrew
brew install git
```

2. Clone the repository and move into it.

```bash
git clone https://github.com/sekika/tffit.git
cd tffit
```

3. Create and activate a virtual environment.

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
```

4. Install the required packages.

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

## Usage

Download the Supplementary Excel file from the [journal article page](https://doi.org/10.1016/j.jenvrad.2026.108077). When downloaded from the journal website, the file may have the following name:

```text
1-s2.0-S0265931X2600192X-mmc1.xlsx
```

Rename it to:

```text
supplementary.xlsx
```

and place it in the current directory, that is, the `tffit/` directory.

## Quick start: Reproducing Tables 4, 5, S3, S4, and S5

To reproduce the main model fitting results, cross-validation results, and supplementary uncertainty analyses, run:

```bash
./fit_supplementary.sh
```

This script outputs:

- Fitted coefficients for all models, corresponding to Table 4.
- In-sample RMSE, R<sup>2</sup>, and SD.
- LOSO RMSE, corresponding to Table 5.
- LOYO RMSE and per-year LOYO RMSE, corresponding to Table 5.
- Coefficient bootstrap confidence intervals, corresponding to Table S3.
- Paired out-of-fold error comparisons, corresponding to Table S4.
- Cluster-level loss differences, corresponding to Table S5.

To save the output to a text file:

```bash
./fit_supplementary.sh > result.txt
```

## Step-by-step commands, manual execution

The analyses can also be run manually with:

```bash
python3 -m src.main
```

The examples below assume that the Supplementary dataset is saved as `supplementary.xlsx` and that the data are in worksheet `"Table S1"`.

### Fit one model on all data

This reproduces the fitted coefficients and in-sample metrics for a selected model.

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model absalom \
  --header-row 3 \
  --data-start 5 \
  --data-end 40
```

### Leave-One-Site-Out, LOSO, cross-validation

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model absalom \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --cv loso
```

### Leave-One-Year-Out, LOYO, cross-validation

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model absalom \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --cv loyo
```

### Generate Table S3: coefficient bootstrap confidence intervals

Table S3 is generated using the selected model and bootstrap resampling of observations.

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model sr1 \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --coef-bootstrap \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --ci-level 0.95 \
  --table-format markdown
```

The output includes:

- Parameter name.
- Estimate from the full dataset.
- Bootstrap mean.
- Bootstrap SD.
- Percentile bootstrap confidence interval.
- Number of successful bootstrap fits.

To save the table to a file:

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model sr1 \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --coef-bootstrap \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --ci-level 0.95 \
  --table-format csv \
  --table-out table_s3.csv
```

To also save individual bootstrap draws:

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model sr1 \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --coef-bootstrap \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --ci-level 0.95 \
  --table-format csv \
  --table-out table_s3.csv \
  --draws-out table_s3_draws.csv
```

### Generate Table S4: paired comparisons of out-of-fold errors

Table S4 compares out-of-fold prediction errors between a base model and other models.

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model sr1 \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --paired-comparison \
  --compare-models absalom k kr krc krp krcs sr2 \
  --validation both \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --ci-level 0.95 \
  --paired-test signflip \
  --table-format markdown
```

The model specified by `--model` is treated as the base model.

The output includes:

- Validation scheme, LOSO or LOYO.
- Model comparison.
- RMSE of the comparison model.
- RMSE of the base model.
- Delta RMSE relative to the base model.
- Bootstrap confidence interval for Delta RMSE.
- Delta MSE relative to the base model.
- Bootstrap confidence interval for Delta MSE.
- Cluster-level paired p-value.

The differences are defined as:

```text
Delta RMSE = RMSE_comparison_model - RMSE_base_model
Delta MSE  = MSE_comparison_model  - MSE_base_model
```

Therefore, positive values indicate lower prediction error for the base model.

For LOSO, sites are resampled as clusters in the paired cluster bootstrap.  
For LOYO, survey years are resampled as clusters.

Because the number of LOYO clusters may be small, the inferential results should be interpreted cautiously.

### Generate Table S5: cluster-level loss differences

Table S5 outputs cluster-level loss differences between a base model and other models.

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model sr1 \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --cluster-loss \
  --compare-models absalom k kr krc krp krcs sr2 \
  --validation both \
  --table-format markdown
```

The cluster-level loss difference is calculated as:

```text
D_c = mean(e_comparison_model^2 - e_base_model^2)
```

where `c` is a site for LOSO or a survey year for LOYO.

Positive values indicate that the base model has lower mean squared error in that cluster.

## Input file requirements

The program reads an Excel worksheet and expects the following columns at a minimum:

- `Year`, integer.
- `Site`, string.
- `TF`, positive.
- `Ex-K`, positive.

Additional columns are required depending on the selected model:

| Model | Additional required columns |
|---|---|
| absalom | `RIP`, `CEC` |
| k | none |
| kr | `RIP` |
| krc | `RIP`, `CEC` |
| krp | `RIP`, `pH` |
| krcs | `RIP`, `137Cs` |
| sr1 | `RIP` |
| sr2 | `RIP` |

Variables used inside logarithms must be positive. Zero or negative values raise explicit errors.

## Command-line options

### Required options

- `--input PATH`  
  Path to the input Excel file.

- `--model NAME`  
  Model name. Available models are:
  `absalom`, `k`, `kr`, `krc`, `krp`, `krcs`, `sr1`, and `sr2`.

- `--sheet NAME`  
  Excel worksheet name to read.

- `--header-row N`  
  Row number containing column headers, using 1-based Excel-style numbering.

- `--data-start N`  
  First row of the data block, using 1-based Excel-style numbering.

- `--data-end N`  
  Last row of the data block, using 1-based Excel-style numbering.

### Fitting and cross-validation options

- `--cv {loso,loyo}`  
  Cross-validation method.  
  If omitted, the model is fit once using all data.

  - `loso`: Leave-One-Site-Out cross-validation.
  - `loyo`: Leave-One-Year-Out cross-validation.

- `--site-col NAME`  
  Column name used to define sites for LOSO.  
  Default: `Site`.

- `--year-col NAME`  
  Column name used to define years for LOYO.  
  Default: `Year`.

- `--exclude-year YEAR [YEAR ...]`  
  Exclude one or more years from LOYO evaluation.

  Example:

  ```bash
  --exclude-year 2016
  ```

- `--data-summary`  
  Print a summary of the loaded data, including column names and sample counts by year.

### Model-specific options

- `--fix-klim`  
  Fix k<sub>lim</sub> to a constant value instead of estimating it.

- `--klim-fixed VALUE`  
  Fixed value of k<sub>lim</sub>.  
  This option is used together with `--fix-klim`.

### Output formatting options

- `--digit N`  
  Number of decimal places for output metrics.  
  Default: `3`.

- `--table-format {text,csv,markdown}`  
  Output format for table-style results.  
  Default: `text`.

- `--table-out PATH`  
  Save table-style output to the specified file.  
  If omitted, the table is printed to standard output.

### Scatter plot options

- `--out PATH`  
  Output path for an observed vs. predicted scatter plot.  
  If omitted, no figure is created.

- `--label LABEL`  
  Custom label displayed in the scatter plot.  
  If omitted, the selected model name is used.

### Coefficient bootstrap options

- `--coef-bootstrap`  
  Run coefficient bootstrap analysis for the selected model.

- `--bootstrap-n N`  
  Number of bootstrap replicates.  
  Default: `10000`.

- `--bootstrap-seed N`  
  Random seed for bootstrap resampling.  
  Default: `12345`.

- `--ci-level VALUE`  
  Confidence interval level.  
  Default: `0.95`.

- `--draws-out PATH`  
  Save individual bootstrap draws to the specified CSV file.  
  This option is used with `--coef-bootstrap`.

### Paired comparison options

- `--paired-comparison`  
  Run paired comparisons of out-of-fold prediction errors between the selected base model and comparison models.

- `--compare-models NAME [NAME ...]`  
  Model names to compare against the selected base model.

- `--validation {loso,loyo,both}`  
  Validation scheme used for paired comparison or cluster-level loss analysis.  
  Default: `both`.

- `--paired-test {signflip,ttest,none}`  
  Cluster-level paired test used in paired comparison analysis.  
  Default: `signflip`.

- `--bootstrap-unit {auto,row,site,year}`  
  Bootstrap resampling unit.  
  This option is retained for interface clarity.  
  Default: `auto`.

### Cluster-level loss options

- `--cluster-loss`  
  Output cluster-level loss differences between the selected base model and comparison models.

## Output

### Standard model fitting

When fitting without `--cv`, the script prints:

- Model equation.
- Fitted parameters.
- In-sample RMSE on the log<sub>10</sub>(TF) scale.
- R<sup>2</sup>.
- Residual SD on the log<sub>10</sub>(TF) scale.

### LOSO cross-validation

When using:

```bash
--cv loso
```

the script prints:

- LOSO micro-averaged RMSE on the log<sub>10</sub>(TF) scale.

### LOYO cross-validation

When using:

```bash
--cv loyo
```

the script prints:

- LOYO micro-averaged RMSE on the log<sub>10</sub>(TF) scale.
- Per-year LOYO RMSE values.

### Supplementary analysis tables

When using `--coef-bootstrap`, `--paired-comparison`, or `--cluster-loss`, the script prints or saves a table in the format specified by `--table-format`.

All RMSE and MSE values are calculated on the log<sub>10</sub>(TF) scale.

## Generating scatter plots

The script can generate an observed vs. predicted scatter plot on the log<sub>10</sub>(TF) scale.

Example:

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model krc \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --out krc.png
```

The plot includes:

- Observed log<sub>10</sub>(TF) values on the x-axis.
- Predicted log<sub>10</sub>(TF) values on the y-axis.
- A dashed 1:1 line.
- Dotted lines representing ±1 residual SD.
- Model label, sample size, and residual SD.
- Square aspect ratio.

A custom label can be specified with:

```bash
--label "Custom label"
```

## Extending the models

This codebase is designed to be extensible.

To add a new model:

1. Create a new file in `src/model/`, for example:

```text
src/model/newmodel.py
```

2. Implement a class inheriting from `BaseModel` with `_fit()` and `_predict()` methods.

3. Register the model using the decorator:

```python
from .registry import register_model

@register_model("newmodel")
class NewModel(BaseModel):
    ...
```

4. Run the model using:

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --sheet "Table S1" \
  --model newmodel \
  --header-row 3 \
  --data-start 5 \
  --data-end 40
```

## Citation

If you use this code in your research, please cite the following paper:

- Seki, K., Yamaguchi, N., Eguchi, T., Igura, M., 2026. Improvement of spatiotemporal generalization in radiocesium transfer models for wheat using symbolic regression. *Journal of Environmental Radioactivity* 298, 108077. https://doi.org/10.1016/j.jenvrad.2026.108077

```bibtex
@article{Seki2026Tffit,
  author  = {Seki, K. and Yamaguchi, N. and Eguchi, T. and Igura, M.},
  title   = {Improvement of spatiotemporal generalization in radiocesium transfer models for wheat using symbolic regression},
  journal = {Journal of Environmental Radioactivity},
  volume  = {298},
  pages   = {108077},
  year    = {2026},
  doi     = {10.1016/j.jenvrad.2026.108077}
}
```

## License

MIT License.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sekika/tffit)
