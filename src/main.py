import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from .data_loader import load_data
from .model.registry import get_model, list_models
from .cross_validation import loso, loyo
from .model.common import log10_strict


def main():
    parser = argparse.ArgumentParser(
        description="Radiocesium Soil-to-Wheat Transfer Models"
    )

    parser.add_argument("--input", required=True, help="Input Excel file")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--cv",
        choices=["loso", "loyo"],
        default=None,
        help="Cross-validation method (optional). If not given, fit on all data."
    )
    parser.add_argument("--sheet", required=True, help="Excel worksheet")
    parser.add_argument("--header-row", type=int, required=True)
    parser.add_argument("--data-start", type=int, required=True)
    parser.add_argument("--data-end", type=int, required=True)

    parser.add_argument("--fix-klim", action="store_true")
    parser.add_argument("--klim-fixed", type=float, default=None)

    parser.add_argument("--exclude-year", type=int, nargs="*", default=None)

    parser.add_argument("--data-summary", action="store_true",
                        help="Display a summary of the data")

    parser.add_argument("--site-col", type=str, default="Site",
                        help="Column name used to define sites")
    parser.add_argument("--year-col", type=str, default="Year",
                        help="Column name used to define years")

    # Add --digit option (Default: 3)
    parser.add_argument("--digit", type=int, default=3,
                        help="Number of decimal places for output metrics (default: 3)")

    # Scatter plot options
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for the scatter plot (e.g., plot.png)")
    parser.add_argument("--label", type=str, default=None,
                        help="Custom label to display in the plot (defaults to model name)")

    args = parser.parse_args()

    try:
        model_class = get_model(args.model)
    except KeyError:
        print(f"Model '{args.model}' not found.")
        print("Available models:")
        for name in list_models():
            print(" -", name)
        sys.exit(1)

    df = load_data(
        file_path=args.input,
        sheet_name=args.sheet,
        header_row=args.header_row,
        data_start=args.data_start,
        data_end=args.data_end
    )

    if args.data_summary:
        print(f"Columns: {', '.join(df.columns.tolist())}")
        counts = df.groupby('Year').size()
        print(" Year: n")
        for year, n in counts.items():
            print(f" {year}: {n}")
        print(f"Total: {len(df)}")

    if "TF" not in df.columns:
        raise KeyError("Column 'TF' not found in the input sheet.")
    if "Ex-K" not in df.columns:
        raise KeyError("Column 'Ex-K' not found in the input sheet.")

    y = log10_strict(df["TF"].to_numpy(), name="TF")
    K = df["Ex-K"].to_numpy(dtype=float)

    model = model_class(fix_klim=args.fix_klim, klim_fixed=args.klim_fixed)

    X = None
    if getattr(model, "features", None):
        missing = [c for c in model.features if c not in df.columns]
        if missing:
            raise KeyError(
                f"Model '{args.model}' requires columns {missing}, but they are missing.")
        X = {c: df[c].to_numpy(dtype=float) for c in model.features}

    # Formatting helper for decimal places
    fmt = f".{args.digit}f"

    if args.cv == "loso":
        rmse = loso(model=model, df=df, site_col=args.site_col)
        print(f"LOSO RMSE: {rmse:{fmt}}")

    elif args.cv == "loyo":
        overall_rmse, per_year_rmse = loyo(
            model=model,
            df=df,
            year_col=args.year_col,
            exclude_years=args.exclude_year
        )
        print(f"LOYO RMSE: {overall_rmse:{fmt}}")

        for y0 in sorted(per_year_rmse.keys()):
            yy = int(y0) % 100
            print(f"  LOYO{yy:02d} RMSE: {per_year_rmse[y0]:{fmt}}")

    else:
        # Fit on all data
        fit_result = model.fit(y, K, X, train_df=df)

        show_keys = ["k1", "k2", "k3", "k4", "k5"]
        if not args.klim_fixed:
            show_keys.append("klim")

        fitted_params = {k: float(v) for k, v in fit_result.items()
                         if k in show_keys and v is not None}

        def _fmt_param(v, sig=4):
            v = float(v)
            return f"{v:.{sig}g}"

        # Parameters use 'g' format (significant figures) based on --digit
        fitted_params_disp = ", ".join(
            f"{k}={_fmt_param(v, sig=args.digit+1)}" for k, v in fitted_params.items())

        y_pred = model.predict(K, X, fit_result)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        r2 = float(r2_score(y, y_pred))
        resid = y_pred - y
        sd = float(np.std(resid, ddof=1))

        print(f'{args.model} model: {model.formula_str}')
        print(fitted_params_disp)
        print(f"RMSE: {rmse:{fmt}}")
        print(f"R^2: {r2:{fmt}}")
        print(f"SD: {sd:{fmt}}")

        # --- Scatter Plot Generation ---
        if args.out:
            plt.figure(figsize=(4.5, 4.5))

            # Determine axis range
            vals = np.concatenate([y, y_pred])
            vals = vals[np.isfinite(vals)]
            vmin, vmax = vals.min(), vals.max()
            pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
            a, b = vmin - pad, vmax + pad

            # Scatter points
            plt.scatter(y, y_pred, s=30, edgecolors='k',
                        linewidths=0.5, color='black', alpha=0.7)

            # 1:1 Reference line
            plt.plot([a, b], [a, b], 'k--', lw=1.0, label='1:1')

            # +/- SD lines (Parallel to 1:1)
            if np.isfinite(sd):
                plt.plot([a, b], [a + sd, b + sd], 'k:', lw=1.0)
                plt.plot([a, b], [a - sd, b - sd], 'k:', lw=1.0)

            # Axis settings
            plt.xlim(a, b)
            plt.ylim(a, b)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('Measured log(TF)')
            plt.ylabel('Predicted log(TF)')

            # Text annotations
            # Use custom label if provided, otherwise capitalize model name
            plot_label = args.label if args.label else args.model.capitalize()
            plt.text(0.05, 0.95, plot_label, transform=plt.gca().transAxes,
                     ha='left', va='top')
            plt.text(0.05, 0.88, f"±SD = {sd:.{args.digit}f}", transform=plt.gca().transAxes,
                     ha='left', va='top')

            plt.tight_layout()
            plt.savefig(args.out)
            print(f"Figure saved to: {args.out}")


if __name__ == "__main__":
    main()
