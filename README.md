# Testing Software for Non-Discrimination: An Updated and Extended Audit in the Italian Car Insurance Domain

This repository contains the data, code, plots, and tables used in the research paper "Testing software for non-discrimination: an updated and extended audit in the Italian car insurance domain". The aim of this research is to test software systems for potential discriminatory practices in the Italian car insurance domain.

## Data
The `data/` directory contains the input dataset used in this research.

## Code
The main scripts are:
- `main.py`: Contains the main code flow, including the import of input data and the call to all the functions;
- `preprocessing.py`: Contains the preprocessing functions;
- `plotting.py`: Contains the code to realize all the figures and plots;
- `discrimination_analysis.py`: Contains the code to compute the results of "RQ1: Do protected attributes (gender, birthplace, age, city, marital status, education, profession) directly influence quoted premiums?".

## Plots
The `plots/` directory contains all the plots and figures in vectorized format used in the paper.

## Tables
The `tables/` directory contains the source of LaTeX tables used in the paper.

## Usage
To replicate the results of the paper, you will need to run the `main.py` script. You will also need to install the required dependencies listed in the `environment.yml` file.
