# Stock-Market-Forecasting-using-LLM
This is a project that leverages machine learning models to predict stock market prices 

## Project Overview
This project demonstrates how Large Language Models (LLMs) can be utilized for stock price forecasting by combining historical data, financial news sentiment analysis, and advanced predictive modeling techniques. The workflow encompasses data preprocessing, model training, inference, and results visualization within an intuitive, reproducible pipeline.

## Data Source
The data used in this project is taken using the CRSP utility from WRDS. One can use SIC Codes to identify the stocks belonging to a particular sector (in this case, finance). It is also required to download the risk-free rate, which is needed to calculate risk-adjusted returns. 

## Prerequisites
- Python == 3.10.11
- Required libraries (install with pip install -r requirements.txt)

## Project Structure
```
Stock-Market-Forecasting-using-LLM/
├── 5-Day/
│   ├── baseline.py
|   ├── ..(additional files)
│   ├── equal_weighted_portfolio.ipynb
|   ├── value_weighted_portfolio.ipynb
|   ├── DM_Test.ipynb
├── ... (additional daily windows)
├── requirements.txt
└── README.md
```
## Running the Forecasting Models
Each Python file in the project represents a day's rolling window implementation. To run any model: 
```
cd N-Day/
python filename
```
## Output Structure
After running any file, the results will be automatically saved in CSV format.

## Diebold-Mariano test
Each window has a Diebold-Mariano test, which is implemented in a Jupyter notebook.

## Portfolio Analysis
Notebooks for portfolio analysis are also given in each directory. All it requires are the prediction CSV files 

## Workflow
1. Data Preparation: Extract data from CRSP/WRDS using SIC codes
2. Model Execution: Run daily rolling window models (python filename.py)
3. Results Generation: Automatic CSV output creation
4. Statistical Testing: Diebold-Mariano tests for model comparison
5. Portfolio Analysis: Performance evaluation using prediction files

## Output Files
```
*results.csv --> Contains the statistical metrics
*predictions.csv --> Contains the predictions along with the actual value. If multiple models are found, results are stored in wide format with each column representing a model.
```
## Key Features 
1. Daily Rolling Windows: Separate implementations for each time window
2. Automated Results Export: CSV generation for easy analysis
3. Comprehensive Testing: Diebold-Mariano statistical validation
4. Portfolio Optimization: Risk-adjusted return analysis
5. Reproducible Research: Self-contained daily implementations

## Tips for Use
1. Ensure all dependencies are installed: pip install -r requirements.txt. TimesFM has strict dependencies 
2. Run models in sequence from day1 to dayN rolling windows
3. Use the portfolio analysis notebooks after generating prediction files
4. Refer to Diebold-Mariano tests for model comparison insights

### Note
All prediction CSV files must be generated before running portfolio analysis or Diebold-Mariano tests, as these analyses depend on the output files from the model implementations.


