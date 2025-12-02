# %% [markdown]
# # Create artifact directory
# We want to create a directory for storing all the artifacts in the current directory. Users can load all the artifacts later for data cleaning pipelines and inferencing.

# %%
# dbutils.widgets.text("Training data max date", "2024-01-31")
# dbutils.widgets.text("Training data min date", "2024-01-01")
# max_date = dbutils.widgets.get("Training data max date")
# min_date = dbutils.widgets.get("Training data min date")

# testnng
max_date = "2024-01-31"
min_date = "2024-01-01"

# %%
import os
import shutil
from pprint import pprint

# shutil.rmtree("./artifacts",ignore_errors=True)
os.makedirs("artifacts",exist_ok=True)
print("Created artifacts directory")

# %% [markdown]
# # Pandas dataframe print options

# %%
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.float_format',lambda x: "%.3f" % x)

# %% [markdown]
# # Helper functions
# 
# * **describe_numeric_col**: Calculates various descriptive stats for a numeric column in a dataframe.
# * **impute_missing_values**: Imputes the mean/median for numeric columns or the mode for other types.

# %%
def describe_numeric_col(x):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x

# %% [markdown]
# # Read data
# 
# We read the latest data from our data lake source. Here we load it locally after having pulled it from DVC.

# %%
!dvc update artifacts/raw_data.csv

# %%
!dvc pull

# %%
print("Loading training data")

data = pd.read_csv("./artifacts/raw_data.csv")

print("Total rows:", data.count())
display(data.head(5))


# %%
import pandas as pd
import datetime
import json

if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()

# Time limit data
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
with open("./artifacts/date_limits.json", "w") as f:
    json.dump(date_limits, f)

# %% [markdown]
# # Feature selection
# 
# Not all columns are relevant for modelling

# %%
data = data.drop(
    [
        "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"
    ],
    axis=1
)

# %%
#Removing columns that will be added back after the EDA
data = data.drop(
    ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
    axis=1
)

# %% [markdown]
# # Data cleaning
# * Remove rows with empty target variable
# * Remove rows with other invalid column data

# %%
import numpy as np

data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)
data["customer_code"].replace("", np.nan, inplace=True)

data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])

data = data[data.source == "signup"]
result=data.lead_indicator.value_counts(normalize = True)

print("Target value counter")
for val, n in zip(result.index, result):
    print(val, ": ", n)
data

# %% [markdown]
# # Create categorical data columns

# %%
vars = [
    "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
]

for col in vars:
    data[col] = data[col].astype("object")
    print(f"Changed {col} to object type")

# %% [markdown]
# # Separate categorical and continuous columns

# %%
cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
cat_vars = data.loc[:, (data.dtypes=="object")]

print("\nContinuous columns: \n")
pprint(list(cont_vars.columns), indent=4)
print("\n Categorical columns: \n")
pprint(list(cat_vars.columns), indent=4)

# %% [markdown]
# # Outliers
# 
# Outliers are data points that significantly differ from the majority of observations in a dataset and can distort statistical analysis or model performance. To identify and remove outliers, one common method is to use the Z-score, which measures how many standard deviations a data point is from the mean. Data points with a Z-score greater than 2 (or sometimes 3) standard deviations away from the mean are typically considered outliers. By applying this threshold, we can filter out values that fall outside the normal range of the data, ensuring that the remaining dataset is more representative and less influenced by extreme values.

# %%
cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
outlier_summary = cont_vars.apply(describe_numeric_col).T
outlier_summary.to_csv('./artifacts/outlier_summary.csv')
outlier_summary

# %% [markdown]
# # Impute data
# 
# In real-world datasets, missing data is a common occurrence due to various factors such as human error, incomplete data collection processes, or system failures. These gaps in the data can hinder analysis and lead to biased results if not properly addressed. Since many analytical and machine learning algorithms require complete data, handling missing values is an essential step in the data preprocessing phase.
# 
# In the next code block, we will handle missing data by performing imputation. For numerical columns, we will replace missing values with the mean or median of the entire column, which provides a reasonable estimate based on the existing data. For categorical columns (object type), we will use the mode, or most frequent value, to fill in missing entries. This approach helps us maintain a complete dataset while ensuring that the imputed values align with the general distribution of each column.

# %%
cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")
cat_missing_impute

# %%
# Continuous variables missing values
cont_vars = cont_vars.apply(impute_missing_values)
cont_vars.apply(describe_numeric_col).T

# %%
cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
cat_vars = cat_vars.apply(impute_missing_values)
cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T
cat_vars

# %% [markdown]
# # Data standardisation
# 
# Standardization, or scaling, becomes necessary when continuous independent variables are measured on different scales, as this can lead to unequal contributions to the analysis. The objective is to rescale these variables so they have comparable ranges and/or variances, ensuring a more balanced influence in the model.

# %%
from sklearn.preprocessing import MinMaxScaler
import joblib

scaler_path = "./artifacts/scaler.pkl"

scaler = MinMaxScaler()
scaler.fit(cont_vars)

joblib.dump(value=scaler, filename=scaler_path)
print("Saved scaler in artifacts")

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
cont_vars

# %% [markdown]
# # Combine data

# %%
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)
print(f"Data cleansed and combined.\nRows: {len(data)}")
data

# %% [markdown]
# # Data drift artifact

# %%
import json

data_columns = list(data.columns)
with open('./artifacts/columns_drift.json','w+') as f:           
    json.dump(data_columns,f)
    
data.to_csv('./artifacts/training_data.csv', index=False)

# %% [markdown] !
# # Binning object columns

# %%
data.columns

# %%
data['bin_source'] = data['source']
values_list = ['li', 'organic','signup','fb']
data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
mapping = {'li' : 'socials', 
           'fb' : 'socials', 
           'organic': 'group1', 
           'signup': 'group1'
           }

data['bin_source'] = data['source'].map(mapping)

# %% [markdown]
# # Save gold medallion dataset

# %%
#spark.sql(f"drop table if exists train_gold")


# %%
# data_gold = spark.createDataFrame(data)
# data_gold.write.saveAsTable('train_gold')
# dbutils.notebook.exit(('training_golden_data',most_recent_date))

data.to_csv('./artifacts/train_data_gold.csv', index=False)
