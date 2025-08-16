# Linear Regression Example

This project demonstrates a simple linear regression using Python, pandas, scikit-learn, and matplotlib.

## Prerequisites

- Python 3.12 or later
- pip (Python package manager)

## Setup

1. (Recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv mlenv
   .\mlenv\Scripts\activate
   ```
2. Install required packages:
   ```powershell
   pip install pandas scikit-learn matplotlib
   ```

## Usage

1. Place your data file `homeprices.csv` in the `linear_regression` folder. The CSV should have columns like `area` and `price`.
2. Run the script:
   ```powershell
   cd linear_regression
   python linear_regression.py
   ```
3. The script will output the predicted price for a given area and display a plot of the regression line.

## Files

- `linear_regression.py`: Main script for training and visualizing the linear regression model.
- `homeprices.csv`: Example data file (not included in repo).

## Notes

- Make sure to activate the virtual environment each time before running the script.
- The `.gitignore` is set up to ignore virtual environments and data files.
