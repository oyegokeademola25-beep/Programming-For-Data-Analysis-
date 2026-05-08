Beijing Air Quality Analysis CMP7005 PRAC1
Cardiff Metropolitan University | School of Technologies | 2025–2026
Student ID: 20318851

This project analyses hourly air quality data collected from four monitoring 
stations in Beijing, China, covering the period March 2013 to February 2017. 
The four stations selected are Dongsi and Tiantan, representing the urban inner-city 
environments, and Shunyi and Huairou representing suburban outer-ring areas. 
The analysis examines spatial and temporal patterns in PM2.5 and other pollutants 
concentrations, with findings connected to prior computational chemistry research 
on metallic oxide carbon monoxide capture by Ademola, Jacob, and Oyegoke (2020).

The work is structured across five tasks. Task 1 involves loading and merging the 
four station datasets into a single cleaned dataset ready for analysis. Task 2 
covers exploratory data analysis, including data understanding, preprocessing, 
seven visualisations, and six summary tables examining seasonal patterns, weekday 
versus weekend differences, and urban-suburban comparisons across all pollutants. 
Task 3 builds a Random Forest regression model to predict hourly PM2.5 
concentrations from co-pollutant and meteorological features, evaluated using 
MAE, RMSE, and R² metrics. Task 4 develops an interactive Streamlit web application 
with four pages covering dataset exploration, visualisations, urban versus suburban 
comparison, and a live PM2.5 predictor with AQI classification. Task 5 documents 
version control using GitHub with descriptive commit messages for each task.

The dataset is sourced from Zhang et al. (2017), Beijing Multi-Site Air Quality 
Data, UCI Machine Learning Repository (https://doi.org/10.24432/C5RK5G). The 
Chinese national air quality standard GB 3095-2012 is applied throughout for 
AQI category thresholds.


LIVE APPLICATION
---------------
The interactive dashboard is deployed and accessible at the following link:
https://epnicm3tmfehbexrf8zqj6.streamlit.app/

IMPORTANT: Before the application can run correctly, you must upload the cleaned 
combined dataset. This file is named all_cities_combined_CLEAN.csv and is included 
in this repository. When you open the application, locate the file uploader in the 
left sidebar and upload this CSV file. The dashboard will not display any data, 
charts, or predictions until the file has been uploaded. Once uploaded, all four 
pages of the application will load correctly, and the full analysis will be 
accessible, including the live PM2.5 predictor.


TO RUN LOCALLY
--------------
Install the dependencies using the following command:
pip3 install streamlit pandas matplotlib seaborn scikit-learn joblib numpy

Then run the application with:
python3 -m streamlit run app.py

Open http://localhost:8501 in your browser and upload the 
all_cities_combined_CLEAN.csv file using the sidebar uploader.
