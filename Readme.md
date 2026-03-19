# tffit: Python implementation of radiocesium transfer factor models for wheat

This repository provides the Python implementation of soil-to-wheat radiocesium (<sup>137</sup>Cs) transfer factor (TF) models analyzed in our accompanying journal article (currently under review). 

It is designed to ensure full transparency and reproducibility by allowing readers and reviewers to execute the model fitting and external cross-validation (LOSO/LOYO) on the Supplementary dataset.

## What this repository is for

1. **Reproducibility:** To allow readers to easily verify the model coefficients and cross-validation results (Table 4 and Table 5) reported in the paper.
2. **Reusability:** To provide a well-structured, reusable codebase for similar soil-to-crop radiocesium transfer analyses, making it straightforward to test alternative model formulations or validation schemes on new datasets.

## Data availability

This repository contains **code only** and does **not** include the dataset used in the paper. The dataset is provided as **Supplementary Material (Table S1)** of the accompanying journal article.

## Features

- **Multiple models supported:** Selectable via the `--model` argument.
- **Target variable:** log<sub>10</sub>(TF)
- **Rigorous Cross-Validation:**
  - **LOSO** (Leave-One-Site-Out) for spatial extrapolation.
  - **LOYO** (Leave-One-Year-Out) for temporal extrapolation, including per-year LOYO RMSE.
- **Robust data handling:** Input variables are processed in linear scale, with strict validation for log transformations (zero or negative values raise explicit errors).

## Models

All equations use log = log<sub>10</sub>.

| Model name | Equation |
|---|---|
| absalom | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(min(K<sub>ex</sub>/CEC, k<sub>lim</sub>) - log(RIP) |
| k       | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) |
| kr      | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) - k<sub>3</sub> log(RIP) |
| krc     | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) + k<sub>3</sub> log(RIP) + k<sub>4</sub> log(CEC) |
| krp     | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) - k<sub>3</sub> log(RIP) - k<sub>4</sub> (pH) |
| krcs    | log(TF) = -k<sub>1</sub> - k<sub>2</sub> log(K<sub>ex</sub>) - k<sub>3</sub> log(RIP) + k<sub>4</sub> log(Cs) |
| sr1     | log(TF) = -(RIP - log(RIP)) - k<sub>1</sub> (k<sub>2</sub> - RIP)K<sub>ex</sub> |
| sr2     | log(TF) = -0.85 RIP - k<sub>1</sub> max(k<sub>2</sub> - RIP, 0)K<sub>ex</sub> |

**Variables and units (as in the paper / Supplementary Table S1):**
- `TF` : transfer factor (dimensionless)
- `Ex-K`: exchangeable potassium (mol/kg)
- `RIP` : radiocesium interception potential (mol/kg)
- `CEC` : cation exchange capacity (molc/kg)
- `137Cs`: radiocesium concentration in soil (Bq/kg)
- `pH` : soil pH (water, 1:5)

Notes:
- K<sub>ex</sub> corresponds to `Ex-K`.
- Cs in ARCs corresponds to the `137Cs` column in the Supplementary file.

## Installation

### For Windows users

* Use [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl/) (WSL) to run this project.

### Setup (Linux, macOS, or WSL)

1. Install git if not already installed:

```bash
# WSL / Ubuntu / Debian
sudo apt update
sudo apt install git -y

# macOS with Homebrew
brew install git
```

2. Clone the repository and navigate into it:

```bash
git clone https://github.com/sekika/tffit.git
cd tffit
```

3. Create and activate a virtual environment:

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
```

4. Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

## Usage

Please download the dataset of the accompanying journal article and save it as `supplementary.xlsx` in the current directory (`tffit/`) before running the scripts.

> **Note for Reviewers/Readers:**
> As the article is currently under review or recently published, please download the Supplementary Excel file from the journal's submission/publication page.

### Quick start: Reproducing Table 4 and Table 5

To reproduce the exact coefficients and validation metrics reported in the paper, simply run the provided bash script:

```bash
./fit_supplementary.sh
```

This script automatically outputs:

* Fitted coefficients (Table 4)
* LOSO RMSE (Table 5)
* LOYO RMSE and per-year LOYO RMSE (Table 5)

To save the output to a text file named `result.txt`, run:

```bash
./fit_supplementary.sh > result.txt
```

### Step-by-step commands (manual execution)

If you wish to run specific evaluations manually, use the `src/main.py` module.

### Fit on all data

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --model absalom \
  --sheet "Table S1" \
  --header-row 3 \
  --data-start 5 \
  --data-end 40
```

### Leave-One-Site-Out (LOSO) cross-validation

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --model absalom \
  --cv loso \
  --sheet "Table S1" \
  --header-row 3 \
  --data-start 5 \
  --data-end 40
```

### Leave-One-Year-Out (LOYO) cross-validation

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --model absalom \
  --cv loyo \
  --sheet "Table S1" \
  --header-row 3 \
  --data-start 5 \
  --data-end 40
```

## Input file requirements (for custom datasets)

The program reads an Excel sheet and expects the following columns at a minimum:

- `Year` (integer)
- `Site` (string)
- `TF` (positive)
- `Ex-K` (positive)

Additional columns are required depending on the chosen model (e.g., `RIP`, `CEC`, `pH`, `137Cs`).

## Command-line options

### Required / core options

- `--input PATH`  
  Path to the Excel file (e.g., `supplementary.xlsx`).

- `--model NAME`  
  Model name. One of:
  `absalom`, `k`, `kr`, `krc`, `krp`, `krcs`, `sr1`, `sr2`.

- `--sheet NAME`
  Excel sheet name to read (e.g., `"Table S1"`).

- `--header-row N`  
  Row number (1-based, Excel-style) containing column headers.

- `--data-start N`
  First row (1-based, Excel-style) of the data block.

- `--data-end N`
  Last row (1-based, Excel-style) of the data block.

### Cross-validation options

- `--cv {loso,loyo}`
  If omitted, the model is fit once using all data (in-sample fit).
  If specified:
  - `loso`: Leave-One-Site-Out cross-validation
  - `loyo`: Leave-One-Year-Out cross-validation (prints overall LOYO RMSE and per-year RMSE)

- `--site-col NAME` (default: `Site`)
  Column name used to define sites (for LOSO).

- `--year-col NAME` (default: `Year`)
  Column name used to define years (for LOYO).

- `--exclude-year YEAR [YEAR ...]`
  Exclude specific year(s) from LOYO evaluation.
  Example: `--exclude-year 2016` (skip LOYO fold for 2016).

### Klim options (Absalom model)

- `--fix-klim`
  Fix k<sub>lim</sub> for absalom model to a constant value instead of estimating it.

- `--klim-fixed VALUE`
  The fixed k<sub>lim</sub> value (mol/kg).

### Output options

- `--digit`
  Number of decimal places for output metrics (default: 3)

- `--data-summary`
  Display a summary of the data. Use this option to verify that the data has been loaded correctly.

## Output

When fitting without `--cv`, the script prints:
- Model equation
- Fitted parameters (k<sub>1</sub>, k<sub>2</sub>, ..., k<sub>lim</sub>)
- In-sample RMSE, R<sup>2</sup> and SD (log10 scale)

When `--cv loso`, it prints:
- LOSO micro-averaged RMSE (log10 scale)

When `--cv loyo`, it prints:
- LOYO micro-averaged RMSE (log10 scale)
- Per-year LOYO RMSE values (e.g., LOYO78, LOYO85, ...)

## Generating scatter plots

The script can generate an observed vs. predicted scatter plot (log<sub>10</sub> scale), similar to Fig. 2 in the acompanying paper, to visualize model performance.

### Usage

To generate a plot, use the `--out` option followed by the desired filename (e.g., `.png`, `.pdf`, or `.svg`).

```bash
python3 -m src.main \
  --input supplementary.xlsx \
  --model krc \
  --sheet "Table S1" \
  --header-row 3 \
  --data-start 5 \
  --data-end 40 \
  --out absalom.png
```

### Plot features

* **1:1 Line:** A dashed line indicating perfect agreement between observed and predicted values.
* **SD Bounds:** Dotted lines representing $\pm 1$ Standard Deviation (SD) of the residuals.
* **Statistics:** The model name and the calculated SD are automatically annotated on the plot.
* **Formatting:** The plot is generated with a fixed aspect ratio (square) to ensure clear comparison of the log-transformed axes.

### Scatter plot options

* `--out PATH`
Specifies the destination path for the plot. If this option is omitted, no image file will be created.
* `--label LABEL`
Custom label to display in the plot.
* `--digit N`
The SD value shown in the plot will respect the number of decimal places specified by this option.

## Extending the models

This codebase is designed to be easily extensible. To add a new model:

1. Create a new file in `src/model/` (e.g., `newmodel.py`).
2. Implement a class inheriting from `BaseModel` with `_fit()` and `_predict()` methods.
3. Register it using the decorator:

```python
from .registry import register_model

@register_model("newmodel")
class NewModel(BaseModel):
    ...
```

4. Run your model using `--model newmodel`.

## Citation

If you use this code in your research, please cite our paper:

> (Citation details will be updated upon publication)

## License

MIT License.
