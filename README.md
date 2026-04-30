# The-Industry-Dimension-of-Shareholder-Wealth-Creation

Step 1: Download the Required Data
You need to download two datasets from external sources. The Kenneth French data file is already included in the repository.
1A. CRSP Monthly Stock File (via WRDS)
Log in to WRDS at wrds-www.wharton.upenn.edu and download the CRSP Monthly Stock File with the following variables:
Variable you need
- PERMNO
- PERMCO
- TICKER
- COMNAM
- RET
- DLRET
- DLSTCD
- PRCP
- SHROUTS
- SHRCD
- EXCHCD
- SICCD

Apply the following filters when downloading:
Date range: January 1990 to December 2024

1B. Compustat North America (via WRDS)
From the same WRDS platform, download the Compustat North America Fundamentals Monthly file with the following variables:
Variables you need
- tic
- sic
  
No additional filters are needed. 
Save the file as:
COMPSTAT_90_24.csv

Step 2: Merge and Prepare the Data
Before running any analysis scripts, the CRSP and Compustat datasets must be merged. This step patches the SIC code 9999, which CRSP assigns to stocks without a verified industry classification, using the earliest available SIC code from Compustat matched by ticker symbol.
Baseline analysis (Q1, Q2, Q3)
Run Merge1.py. This produces the cleaned and patched dataset used for the main analysis:
Output file: CRSP_90_24_patched.csv

Run Merge2.py for Robustness. This produces the cleaned and patched dataset with Shumway Delisting Returns:
Output file: CRSP_90_24_patched_shumway.csv

Step 3: Run the Analysis
Once the merged files are in place, the analysis scripts can be run. Each script reads the relevant merged file directly and produces all tables and figures for the corresponding section of the thesis.

Optional diagnostic
If you want to inspect which stocks carry SIC code 9999 before patching, run:
This prints the 50 largest unclassified stocks by average market capitalisation and requires only the raw CRSP_90_24.csv file.

Setup
The scripts are written in Python 3 and require the following packages:
pip install pandas numpy scipy matplotlib
