# Stock-Market-Forecasting-using-LLM
This is a project that leverages machine learning models to predict stock market prices 

## Project Overview
This project demonstrates how Large Language Models (LLMs) can be utilized for stock price forecasting by combining historical data, financial news sentiment analysis, and advanced predictive modeling techniques. The workflow encompasses data preprocessing, model training, inference, and results visualization within an intuitive, reproducible pipeline.

## Data
The data used in this project is taken using the CRSP utility from WRDS. One can use SIC Codes to identify the stocks belonging to a particular sector (in this case, finance). It is also required to download the risk-free rate, which is needed to calculate risk-adjusted returns. 

## Prerequisites
- Python == 3.10.11
- Required libraries (install with pip install -r requirements.txt)

## Running the Forecasting Models
Each Python file in the project represents a day's rolling window implementation. To run any model, go to each directory and run:
python filename

## Output Structure
After running any file, the results will be automatically saved in CSV format.

## Diebold-Mariano test
Each window has a Diebold-Mariano test, which is implemented in a Jupyter notebook.

## Portfolio Analysis
Notebooks for portfolio analysis are also given in each directory. All it requires are the prediction CSV files 
