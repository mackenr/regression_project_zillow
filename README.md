Zillow Project
----

## Objectives:
- Document the data science pipeline. Ensure the findings are presented clearly and the documentation is clear enough for independent reproduction
- Create modules that can be downloaded for the sake of reproducibility.






## Project Description and goals:

- The goal is to use data to find and explore predictive factors of home value.
- Ultimately we hope to use these factors to accurateley asses home values for zillow.

## Questions:

Questions:

Generally I am interested in if time or location is significant to consider in this data. The specific questions we ask aboout this are asked in the mvp.ipynb file.









## Data Dictionary:
Note: the target variable is taxvaluedollarcnt


|Name|                                    Dtype|
|-------------|-------------|
|area|                                    float64|
|assessmentyear|                          float64|
|bathdividesarea|                         float64|
|bathdividesbed|                          float64|
|bathplusbathdividesarea|                 float64|
|bathroomcnt|                             float64|
|bed|+bath*area                           float64|
|bedbath_harmean|*area                    float64|
|beddivdesbath|                           float64|
|bedroomcnt|                              float64|
|calculatedbathnbr|                       float64|
|county|                                   object|
|decade|                                    int64|
|fips|                                      int64|
|fullbathcnt|                             float64|
|id|                                        int64|
|latitude|                                float64|
|logerror|                                float64|
|longitude|                               float64|
|lotsizesquarefeet|                       float64|
|lotsizesquarefeet_wo_outliers|           float64|
|roomcnt|                                 float64|
|sqrt(bed^2+bath^2)*area|                 float64|
|taxvaluedollarcnt|                       float64|
|transaction_month|                        object|
|transaction_month_int|                     int64|
|transactiondate|                  datetime64[ns]|
|transactiondate_in_days|                   int64|
|yearbuilt|                               float64|




## Procedure:

#### Planning:
Overview. Hypothesis Etc

#### Acquisition:
An wrangele.py file is created and used. It aquires the data from the database then saves it a .csv file locally (telco.csv). Also it outputs simple graphs of the counts of unique values per variable in order to give a quick visual of whether or not the data is going to be categorical or not.

#### Preparation:
This is lumped into the wrangle.py

#### Exploration and Pre-processing:
We do our split once in order to prevent "data poisoning".
From here we start doing a deep dive into exploration on the train dataset. We ask questions of our data and create graphs in order to better understand our data and ask better questions. We then formulate those questions into hypothesises and do some statisitical tests to find the answers to our questions. 

#### Modeling:
Here we use select various machine learning algorithms from the sklean library to create models. Once we have our models we can further vary our hyperparmeters in each model. From here we 

#### Delivery:
A final report is created which gives a highlevel overview of the the process. 

## Explanations for Reproducibility:
 
In order to repoduce these you will need a env.py file which contains host, username and password creditials to access the sql server. You will also need to create a califips.csv file for the fips data in california. The data had a leading zero removed so I just did the same with leading zero.
infact all the data you need is the following:
fips_code,county
6059, Orange County
6037, Los Angeles County

The remaining files are availble within my github repo. If you clone this repo, add a env.py file in the format shown below you will be able to reproduce the outcome. As and aside the random state is included in the file. If you were to change this your results my slightly differ.

```python

host='xxxxx'
username='xxxxxx'
password='xxxxxx'
## Where the strings are your respective credentials

```

## Executive Summary:

#### Conclusion:
Conclusion:
Magie can rest easy:
Our data all comes from LA or Orange county in California.
Used FIPS to find the county’s and states.
 
 
 
Our selected top models all beat or matched baseline. It is worth further exploring hyperparameters and the partitioned data. Also to note there are model selection criteria we could have used. As such it would be worth investigating Mean Absolute Error(MAE) and $ R^2 $ as model selection criteria. Perhaps one model would perform well in each and might be a better general model.
 
Note: There are models that generalize well to all the data and currently the combined data model has better performance than the separated. I think this warrants further investigation as shown by the statistical inquires and simply that our data is roughly 3/4 from LA and 1/4 from OC. Ultimately we would wish to generalize this proves and find the optimal scale : state, county
,zip, zones based on a measure of central tendency and population density etc. Since we have geodata this is all possible but will take detailed and creative exploratory analysis. 

#### Specific Recommendations:
It would be nice to exlore the data partitioned by county in more detail.

#### Actionable Example:
I would like to spend more time creating combinations of variables that predict our taxvaluedollarcnt for the data partitioned by county.




#### Closing Quote:
>
>“Errors using inadequate data are much less than those using no data at all.”
(Charles Babbage, English Mathematician)



