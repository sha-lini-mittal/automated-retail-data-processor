Automated Retail Data Processor

Automated Retail Data Processor - A Streamlit-based pipeline to clean retail transaction data, produce aggregated monthly/weekly datasets, run EDA, and generate simple moving-average (SMA) forecasts for quantity, price and revenue.

Short description: Automated pipeline for cleaning retail sales data, producing aggregated time-series, interactive dashboards and SMA-based forecasts with evaluation metrics and downloadable outputs.

Highlights:

Automated data cleaning & aggregation (monthly & weekly).

Interactive Streamlit UI for EDA, downloads and pipeline control. See sales.py. 

sales

SMA forecasting dashboard with evaluation (MAE, RMSE, MAPE, R²), confidence intervals and downloadable forecasts. See sales_forecast.py. 

sales_forecast

Export processed files (cleaned, monthly & weekly aggregates) for BI or ML use.

Suggested repository name & short description:

Repo name: automated-retail-data-processor

Short description: Streamlit pipeline to clean retail data, aggregate sales, and forecast using Simple Moving Average (SMA).

Project structure:

automated-retail-data-processor/
│
├── sales.py                  # Main Streamlit app: ingestion, cleaning, aggregation, downloads. :contentReference[oaicite:2]{index=2}
├── sales_forecast.py         # Streamlit forecasting dashboard (SMA + evaluation). :contentReference[oaicite:3]{index=3}
├── requirements.txt          # Python dependencies
├── README.md                 # Project docs (this file)
├── datasets/                 # Put input CSVs and generated CSVs here
│   ├── online_retail.csv
│   ├── cleaned_data_with_filled_descriptions.csv
│   ├── monthly_sales_aggregated.csv
│   └── weekly_sales_aggregated.csv
└── notebooks/                # Optional: Jupyter notebooks, experiments

Installation & local setup:

Clone the repo:

git clone https://github.com/YOUR_USERNAME/automated-retail-data-processor.git
cd automated-retail-data-processor

Create and activate a virtual environment (recommended):

Windows (PowerShell)

python -m venv venv
venv\Scripts\Activate.ps1   # or venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Suggested requirements.txt

streamlit
pandas
numpy
plotly
matplotlib
scikit-learn
joblib
openpyxl


(Adjust versions as needed. The apps use Streamlit, pandas, numpy and plotly heavily.) 

sales_forecast

Run the apps locally:

Data processing / EDA / Downloads

streamlit run sales.py


This app handles ingestion, cleaning, filling missing descriptions, aggregation and download of processed CSVs. 

sales

Forecasting dashboard (SMA)

streamlit run sales_forecast.py


Provides SMA forecasting controls (window size, forecast periods), evaluation metrics and downloadable forecasts. 

sales_forecast

Open http://localhost:8501 in your browser (Streamlit usually opens automatically).

How to use (quick flow):

Place your raw retail CSV (e.g., online_retail.csv) into datasets/ or upload via the sales.py app. 

sales

Run sales.py — execute the automated processing pipeline to produce:

cleaned_data_with_filled_descriptions.csv

monthly_sales_aggregated.csv

weekly_sales_aggregated.csv
These become the inputs for the forecasting app. 

sales

Run sales_forecast.py to explore products, switch monthly/weekly views, enable SMA forecasting, tune window size and forecast horizon, evaluate model and download forecasts. 

sales_forecast

Forecasting & model details:

Algorithm: Simple Moving Average (SMA) forecasting — forecast equals mean of the last window_size historical periods. Implemented in simple_moving_average_forecast inside sales_forecast.py.sales_forecast

Controls: window size (2–10), forecast periods (1–12), switch between monthly/weekly aggregation. 

sales_forecast

Evaluation: Time-aware train/test split and metrics computed (MAE, MSE, RMSE, MAPE, R²). See calculate_evaluation_metrics in sales_forecast.py. 

sales_forecast

Confidence intervals: approximate 95% CIs via rolling-standard-deviation based method (implemented). 

sales_forecast

Expected input columns (typical retail dataset):

The processing pipeline expects fields similar to the UCI Online Retail dataset:

InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice (or Price), CustomerID, Country
Sales.py code inspects and aggregates columns into monthly/weekly summaries — check the script if your column names differ. 

sales

Troubleshooting & tips:
If python --version fails, make sure you typed it correctly (no space between -- and version): python --version (Typing python -- version will attempt to open a file called version in the current directory.)
If you see warnings about scripts installed to a Scripts directory not on PATH (e.g., watchmedo.exe, dotenv.exe), either: add that Scripts path to your PATH, or use the full path to the script, or activate your virtual environment (recommended) where the Scripts directory will be on PATH while active.
If the app complains missing CSVs, confirm processed files are in datasets/ or re-run sales.py to create them. 

Contributing:
Fork the repository.
Create a feature branch: git checkout -b feature/your-feature.
Make changes, commit: git commit -am "Add feature".
Push and open a Pull Request.
(Include tests or notebooks in /notebooks when possible.)

License: MIT License - feel free to reuse and modify. Add your LICENSE file with MIT text.

Future enhancements (ideas):
Add more forecasting algorithms (ARIMA, Prophet, LSTM).
Implement backtesting and more robust cross-validation for time-series.
Add notifications/alerts for stock-outs or sudden revenue drops.
Build a REST API endpoint to serve forecasts.
Add automated unit tests and CI pipeline.
