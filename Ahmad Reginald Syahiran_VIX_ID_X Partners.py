#!/usr/bin/env python
# coding: utf-8

# # **Evaluation of Credit Risk**
# * Created by Ahmad Reginald Syahiran
# * Presented as my capstone project for VIX at ID/X Partners

# # **Understanding the Business Context**

# * The concept of credit risk revolves around the possibility of a borrower not repaying a loan
# * Therefore, evaluating the likelihood of repayment by the borrower is vital in managing credit risk
# * Machine learning can be utilized to streamline this evaluation process"

# # **Strategic Analysis Methodology**

# * Analysis through description
# * Analysis using graphical representations
# * Predictive modeling via classification techniques

# # **Data Requirement**

# Dataset of customer loan from financial company

# # **Data Collection**
# 
# Dataset is collected by ID/X Partners from a company

# In[ ]:


#Mount Drive
from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


df_cred = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/loan_data_2007_2014.csv')
df_cred.head(2)


# # **Data Understanding**

# In[ ]:


df_cred.info()


# ## Missing Values

# In[ ]:


full_non_null = [col for col in df_cred.columns if df_cred[col].isnull().all()]
print(full_non_null)
print(len(full_non_null),"column(s)")


# * Identified 17 columns with complete data, suggesting their removal

# In[ ]:


df_cred = df_cred.drop(axis=1, columns=full_non_null)
df_cred.info()


# In[ ]:


percent_missing = df_cred.isnull().sum() * 100 / len(df_cred)
dtypes=[df_cred[col].dtype for col in df_cred.columns]
missing_value_df = pd.DataFrame({'data_type':dtypes,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', ascending=False, inplace=True)
missing_value_df.head(10)


# ### Data Absence Exceeding 50%:
# 
# * mths_since_last_record: Duration in months since the last noted public record.
# * mths_since_last_major_derog: Time in months since the last significant derogatory mark, such as a 90-day or more severe credit rating.
# * desc: Description of the loan as given by the loan applicant.
# * mths_since_last_delinq: Interval in months since the applicant's most recent missed payment.
# 
# ### Data Absence Between 40% and 50%:
# 
# * next_payment_d: The most recent month in which a loan payment was made.
# ### Data Absence Ranging from 1% to 20%:
# 
# * tot_cur_bal: Aggregate balance across all accounts at present.
# * tot_coll_amt: Cumulative amount ever owed in collections.
# * total_rev_hi_lim: Maximum limit of all revolving credit lines.
# * emp_title: Job designation provided by the loan applicant.
# * emp_length: Duration of employment, ranging from 0 (less than a year) to 10 (a decade or more).
# 

# In[ ]:


missing_value_df.tail(48)


# Missing values below 1%:
# 
# * last_pymnt_d: The most recent month in which a payment was received.
# * revol_util: The rate of usage of revolving credit lines compared to the total available credit.
# * collections_12_mths_ex_med: Count of non-medical collections in the past 12 months.
# * last_credit_pull_d: The latest month when the lender conducted a credit check for this loan.
# * pub_rec: Tally of negative public records.
# * inq_last_6mths: Inquiries in the past six months.
# * delinq_2yrs: Instances of delinquency in the past two years.
# * open_acc: Number of active credit lines in the borrower's credit report.
# * earliest_cr_line: The opening month of the borrower's first credit line as reported.
# * acc_now_delinq: Number of accounts where the borrower is currently behind in payments.
# * total_acc: Total count of credit lines in the borrower's credit history.
# * title	object

# ## Duplicated Data

# In[ ]:


df_cred.duplicated().sum()


# * There is no duplicated data in hthe dataset

# # **Data Preparation**

# ## Handling Missing Values

# ### Removing `mths_since_last_record`, `desc`, and `next_pymnt_d`

# In[ ]:


df_cred = df_cred[df_cred.columns[~df_cred.columns.isin(['mths_since_last_record','desc','next_pymnt_d'])]]
df_cred.info()


# In[ ]:


df_cred.head(2)


# ### Imputation

# * For `mths_since_last_major_derog` and `mths_since_last_delinq` (missing values above 50%), I imputed them with "0" (zero) value
# * For others, I used its median value for numerical features and mode for categorical features

# In[ ]:


for col in ['mths_since_last_major_derog','mths_since_last_delinq']:
    df_cred[col] = df_cred[col].fillna(0)


# In[ ]:


df_cred[['mths_since_last_major_derog','mths_since_last_delinq']].isnull().sum()


# In[ ]:


# Numerical columns
for col in df_cred.select_dtypes(exclude='object'):
    df_cred[col] = df_cred[col].fillna(df_cred[col].median())
df_cred.isnull().sum()


# In[ ]:


# Non numerical columns
for col in df_cred.select_dtypes(include='object'):
    df_cred[col] = df_cred[col].fillna(df_cred[col].mode().iloc[0])
print("Updated Missing Values")
df_cred.isnull().sum()


# In[ ]:


df_cred.head(3)


# ## Checking Unique Values

# In[ ]:


print("Unique Features (Numerical)")
print(df_cred.select_dtypes(exclude='object').nunique())


# * `Unnamed: 0`, `id`, and `member_id` are unique  towards each of rows
# * `policy_code` have only single unique value

# In[ ]:


print("Unique Features (Categorical)")
print(df_cred.select_dtypes(exclude=['int','float']).nunique())


# * `emp_title`, `url`,`title`, `zip_code`, `earliest_cr_line` has more than 500 unique values
# * `last_credit_pull_d`,`last_pymnt_d`,`issue_d`,`addr_state` have at least 50 unique values (below 500)
# * `application_type` only have single unique value

# In[ ]:


df_cred["term"].unique()


# * I need to clean the whitespace

# In[ ]:


def word_strip(x):
  return x.strip()

df_cred['term'] = df_cred['term'].apply(lambda x: word_strip(x))
df_cred["term"].unique()


# In[ ]:


df_cred["grade"].unique()


# In[ ]:


df_cred["sub_grade"].unique()


# In[ ]:


df_cred["emp_length"].unique()


# In[ ]:


df_cred["home_ownership"].unique()


# In[ ]:


df_cred["verification_status"].unique()


# In[ ]:


df_cred["purpose"].unique()


# ## Formatting Target Variable

# * `loan_status` would be our target variable
# * However, I can not implement it directly on 9 unique values
# * I will group them into group for binary classification

# In[ ]:


df_cred["loan_status"].unique()


# * Good Loan (1) : `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`
# * Bad Loan (0) : `Charged Off`, `Does not meet the credit policy. Status:Charged Off`, `Default`,  `Late (31-120 days)`
# * Undetachable Loan (-1) : `Current`, `In Grace Period`, `Late (16-30 days)`
# * I will use the Good Loan (1) and the Bad Loan (0) later for binary classification
# * Later, Undetachable Loan (-1) columns will be dropped because it is still current loan in progress that can not be detected as good or bad

# In[ ]:


# Define a dictionary for encoding target variable
target_dict = {'Fully Paid':1,
               'Does not meet the credit policy. Status:Fully Paid':1,
               'Charged Off':0,
               'Does not meet the credit policy. Status:Charged Off':0,
               'Default':0,
               'Late (31-120 days)':0,
               'Current':-1,
               'In Grace Period':-1,
               'Late (16-30 days)':-1}
# Create the mapped values in a new column
df_cred['loan_status'] = df_cred['loan_status'].map(target_dict)
# Review dataset
df_cred.head()


# In[ ]:


df_cred = df_cred.loc[~df_cred['loan_status'].isin([-1])].reset_index(drop=True)
df_cred.info()


# In[ ]:


df_cred.tail()


# ## Datetime setting

# In[ ]:


# The month the borrower's earliest reported credit line was opened
df_cred['earliest_cr_line'].value_counts()


# In[ ]:


df_cred['earliest_cr_line'] = pd.to_datetime(df_cred['earliest_cr_line'], format='%b-%y')


# In[ ]:


# The most recent month LC pulled credit for this loan
df_cred['last_credit_pull_d'].value_counts()


# In[ ]:


# Last month payment was received
df_cred['last_pymnt_d'].value_counts()


# In[ ]:


# The month which the loan was funded
df_cred['issue_d'].value_counts()


# In[ ]:


df_cred[['issue_d','last_pymnt_d','last_credit_pull_d']].head(3)


# In[ ]:


def date_time(dt):
  if dt.year > 2016:
    dt = dt.replace(year=dt.year-100)
  return dt


# In[ ]:


# Set standard datetime
df_cred['earliest_cr_line'] = pd.to_datetime(df_cred['earliest_cr_line'], format='%b-%y') # The month the borrower's earliest reported credit line was opened
df_cred['earliest_cr_line'] = df_cred['earliest_cr_line'].apply(lambda x: date_time(x))
df_cred['issue_d'] = pd.to_datetime(df_cred['issue_d'], format='%b-%y') # The month which the loan was funded
df_cred['last_pymnt_d'] = pd.to_datetime(df_cred['last_pymnt_d'],format='%b-%y') # Last month payment was received
df_cred['last_credit_pull_d'] = pd.to_datetime(df_cred['last_credit_pull_d'],format='%b-%y') # The most recent month LC pulled credit for this loan
df_cred[['earliest_cr_line','issue_d','last_pymnt_d','last_credit_pull_d']].head(3)


# I created a new column for datetime:
# * `pymnt_time` = the number of months between funded loan (`issue_d`) and last received payment (`last_pymnt_d`)
# * `credit_pull_year` = the number of years between borrower's earliest reported credit line was opened (`earliest_cr_line`) and the most recent LC pulled credit for this loan (`last_credit_pull_d`)

# In[ ]:


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


# In[ ]:


def diff_year(d1, d2):
    return (d1.year - d2.year)


# In[ ]:


((df_cred.apply(lambda x: diff_month(x.last_pymnt_d, x.issue_d), axis=1) < 0)).any().any()


# In[ ]:


((df_cred.apply(lambda x: diff_month(x.last_credit_pull_d, x.earliest_cr_line), axis=1) < 0)).any().any()


# In[ ]:


df_cred['pymnt_time'] = df_cred.apply(lambda x: diff_month(x.last_pymnt_d, x.issue_d), axis=1)
df_cred['credit_pull_year'] = df_cred.apply(lambda x: diff_year(x.last_credit_pull_d, x.earliest_cr_line), axis=1)
print('Adding features succeed')


# In[ ]:


df_cred.info()


# In[ ]:


df_cred.head(3)


# In[ ]:


df_cred.to_csv('df_cred.csv', index=False)
get_ipython().system("cp 'df_cred.csv' '/content/drive/MyDrive/Credit Risk Assessment'")
print('Saving cleaned data is done!')


# * This cleaned data will be analysed in Exploratory Data Analysis for extracting insights
# * Now, I am going to prepare for predictive modelling

# In[ ]:


df_cred = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/df_cred.csv')
df_cred.head(2)


# ## Analysing Descriptive Statistics

# ### Numerical Features

# In[ ]:


df_cred.describe()


# ### Categorical Features

# In[ ]:


df_cred.describe(exclude=['int','float'])


# ## Analysing Distribution Plot

# In[ ]:


df_cred.dtypes.value_counts()


# In[ ]:


non_used = ['Unnamed: 0','id','member_id','policy_code', 'loan_status']
uni_dist = df_cred.select_dtypes(include=[np.float64,np.int64])
uni_dist = uni_dist[uni_dist.columns[~uni_dist.columns.isin(non_used)]]


# In[ ]:


plt.figure(figsize=(40, 20))
for i in range(0, 11):
    plt.subplot(11, 3, i+1)
    sns.distplot(uni_dist.iloc[:,i], color='red')
    plt.tight_layout()


# In[ ]:


plt.figure(figsize=(40, 20))
for i in range(11, 22):
    plt.subplot(11, 3, i+1)
    sns.distplot(uni_dist.iloc[:,i], color='red')
    plt.tight_layout()


# In[ ]:


plt.figure(figsize=(40, 20))
for i in range(22, 33):
    plt.subplot(12, 3, i+1)
    sns.distplot(uni_dist.iloc[:,i], color='red')
    plt.tight_layout()


# * Most of the features are skewed
# * Non skewed features: `loan_amnt`,`funded_amnt`,`funded_amnt_inv`,`int_rate`,`dti`

# ## Analysing Box Plot

# In[ ]:


plt.figure(figsize=(40, 20))
for i in range(0, 11):
    plt.subplot(11, 3, i+1)
    sns.boxplot(uni_dist.iloc[:,i], color='red',orient='v')
    plt.tight_layout()


# In[ ]:


plt.figure(figsize=(40, 20))
for i in range(11, 22):
    plt.subplot(11, 3, i+1)
    sns.boxplot(uni_dist.iloc[:,i], color='red',orient='v')
    plt.tight_layout()


# In[ ]:


plt.figure(figsize=(40, 20))
for i in range(22, 33):
    plt.subplot(12, 3, i+1)
    sns.boxplot(uni_dist.iloc[:,i], color='red',orient='v')
    plt.tight_layout()


# * Most of the features have outliers
# * Features with no outliers: `loan_amnt`,`funded_amnt`,`funded_amnt_inv`,`int_rate`

# Severe outliers:
# * installment
# * annual_inc
# * open_acc
# * revol_bal
# * total_pymnt_inv
# * out_prncp
# * total_rec_late_fee
# * out_prncp_inv
# * total_rec_prncp
# * total_pymnt
# * total_acc
# * total_rec_int
# * last_pymnt_amnt
# * total_rev_hi_lim
# * recoveries
# * total_coll_amt
# * pymnt_time
# * collection_recovery_fee
# * tot_cur_bal
# * credit_pull_year

# ## Correlation Analysis for Feature Selection

# In[ ]:


non_used = ['Unnamed: 0','id','member_id','policy_code','loan_status']
uni_dist = df_cred.select_dtypes(include=[np.float64,np.int64])
uni_dist = uni_dist[uni_dist.columns[~uni_dist.columns.isin(non_used)]]
fig = plt.figure(figsize = (40,10))
sns.heatmap(uni_dist.corr(),cmap='Reds', annot = True);


# In[ ]:


def top_correlation (df,n):
    corr_matrix = df.corr()
    correlation = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
    correlation = pd.DataFrame(correlation).reset_index()
    correlation.columns=["Variable_1","Variable_2","Correlation"]
    correlation = correlation.reindex(correlation.Correlation.abs().sort_values(ascending=False).index).reset_index().drop(["index"],axis=1)
    return correlation.head(n)
print("High Correlated Features (Corr > 0.5)")
top_correlation(uni_dist,39)


# In[ ]:


df_corr = top_correlation(uni_dist,41)
df_corr.to_excel('df_corr_3.xlsx', index=False)
get_ipython().system("cp 'df_corr_3.xlsx' '/content/drive/MyDrive/Credit Risk Assessment'")
print('Saving correlation data is done!')


# In[ ]:


uni_dist.head()


# * `emp_title`, `url`,`title`, `zip_code`, `earliest_cr_line` has more than 500 unique values
# * `last_credit_pull_d`,`last_pymnt_d`,`issue_d`,`addr_state` have at least 50 unique values (below 500)
# * `application_type` only have single unique value

# In[ ]:


removed_unused = ['Unnamed: 0','id','member_id','policy_code','emp_title','url','title','zip_code','earliest_cr_line']
multicol = ['last_credit_pull_d','last_pymnt_d','issue_d','addr_state','application_type',
            'out_prncp_inv','funded_amnt','total_pymnt_inv','funded_amnt_inv','total_rec_prncp','out_prncp',
            'revol_bal','total_pymnt','recoveries','total_rec_int','total_acc','loan_amnt']
removed_all = removed_unused + multicol


# In[ ]:


df_cred_a = df_cred[df_cred.columns[~df_cred.columns.isin(removed_all)]].reset_index(drop=True)
df_cred_b = df_cred[df_cred.columns[~df_cred.columns.isin(removed_unused)]].reset_index(drop=True)


# In[ ]:


df_cred_a.head()


# In[ ]:


df_cred_b.head()


# In[ ]:


df_cred_a.to_csv('df_cred_a.csv', index=False)
get_ipython().system("cp 'df_cred_a.csv' '/content/drive/MyDrive/Credit Risk Assessment'")
print('Saving cleaned data is done!')

df_cred_b.to_csv('df_cred_b.csv', index=False)
get_ipython().system("cp 'df_cred_b.csv' '/content/drive/MyDrive/Credit Risk Assessment'")
print('Saving cleaned data is done!')


# ## Categorical Encoding

# In[ ]:


df_cred_a = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/df_cred_a.csv')
df_cred_a.head(2)


# In[ ]:


df_cred_a.info()


# In[ ]:


df_cred_b = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/df_cred_b.csv')
df_cred_b.head(2)


# In[ ]:


df_cred_a["term"].unique()


# In[ ]:


def text_num(text):
  return [int(s) for s in text.split() if s.isdigit()][0]


# In[ ]:


sns.distplot(df_cred_a["term"].apply(lambda x: text_num(x)), color='red')
plt.tight_layout()
plt.show()


# In[ ]:


df_cred_a["term"] = df_cred_a["term"].apply(lambda x: text_num(x))
df_cred_a.head()


# In[ ]:


df_cred_a["grade"].unique()


# In[ ]:


# Define a dictionary for encoding ordinal variable
target_dict = {'A':6,
               'B':5,
               'C':4,
               'D':3,
               'E':2,
               'F':1,
               'G':0}
# Create the mapped values in a new column
df_cred_a["grade"] = df_cred_a["grade"].map(target_dict)


# In[ ]:


# Review dataset
df_cred_a.head()


# In[ ]:


df_cred_a["sub_grade"].unique()


# In[ ]:


def f_A(row):
    if row == 'A1':
        val = 1
    elif row == 'A2':
        val = 2
    elif row == 'A3':
        val = 3
    elif row == 'A4':
        val = 4
    elif row == 'A5':
        val = 5
    else:
        val = 0
    return val

def f_B(row):
    if row == 'B1':
        val = 1
    elif row == 'B2':
        val = 2
    elif row == 'B3':
        val = 3
    elif row == 'B4':
        val = 4
    elif row == 'B5':
        val = 5
    else:
        val = 0
    return val

def f_C(row):
    if row == 'C1':
        val = 1
    elif row == 'C2':
        val = 2
    elif row == 'C3':
        val = 3
    elif row == 'C4':
        val = 4
    elif row == 'C5':
        val = 5
    else:
        val = 0
    return val

def f_D(row):
    if row == 'D1':
        val = 1
    elif row == 'D2':
        val = 2
    elif row == 'D3':
        val = 3
    elif row == 'D4':
        val = 4
    elif row == 'D5':
        val = 5
    else:
        val = 0
    return val

def f_E(row):
    if row == 'E1':
        val = 1
    elif row == 'E2':
        val = 2
    elif row == 'E3':
        val = 3
    elif row == 'E4':
        val = 4
    elif row == 'E5':
        val = 5
    else:
        val = 0
    return val

def f_F(row):
    if row == 'F1':
        val = 1
    elif row == 'F2':
        val = 2
    elif row == 'F3':
        val = 3
    elif row == 'F4':
        val = 4
    elif row == 'F5':
        val = 5
    else:
        val = 0
    return val

def f_G(row):
    if row == 'G1':
        val = 1
    elif row == 'G2':
        val = 2
    elif row == 'G3':
        val = 3
    elif row == 'G4':
        val = 4
    elif row == 'G5':
        val = 5
    else:
        val = 0
    return val


# In[ ]:


df_cred_a['SubGrade_A'] = df_cred_a["sub_grade"].apply(f_A)
df_cred_a['SubGrade_B'] = df_cred_a["sub_grade"].apply(f_B)
df_cred_a['SubGrade_C'] = df_cred_a["sub_grade"].apply(f_C)
df_cred_a['SubGrade_D'] = df_cred_a["sub_grade"].apply(f_D)
df_cred_a['SubGrade_E'] = df_cred_a["sub_grade"].apply(f_E)
df_cred_a['SubGrade_F'] = df_cred_a["sub_grade"].apply(f_F)
df_cred_a['SubGrade_G'] = df_cred_a["sub_grade"].apply(f_G)
df_cred_a = df_cred_a.drop(axis=1, columns="sub_grade")


# In[ ]:


df_cred_a.head()


# In[ ]:


df_cred_a["emp_length"].unique()


# In[ ]:


# Define a dictionary for encoding ordinal variable
target_dict = {'< 1 year':0,
               '1 year':1,
               '2 years':2,
               '3 years':3,
               '4 years':4,
               '5 years':5,
               '6 years':6,
               '7 years':7,
               '8 years':8,
               '9 years':9,
               '10+ years':10}
# Create the mapped values in a new column
df_cred_a["emp_length"] = df_cred_a["emp_length"].map(target_dict)


# In[ ]:


df_cred_a.head()


# In[ ]:


df_cred_a["home_ownership"].unique()


# In[ ]:


df_cred_a["home_ownership"].value_counts()


# * ANY, OTHER, and NONE will be aggregated into OTHER
# * Then I used One-Hot-Encoding for this feature

# In[ ]:


# Define a dictionary for aggregating variable
target_dict = {'MORTGAGE':'MORTGAGE',
               'RENT':'RENT',
               'OWN':'OWN',
               'OTHER':'OTHER',
               'ANY':'OTHER',
               'NONE':'OTHER'}
# Create the mapped values in a new column
df_cred_a["home_ownership"] = df_cred_a["home_ownership"].map(target_dict)


# In[ ]:


df_cred_a.head()


# In[ ]:


encoder = OneHotEncoder(sparse=False)
df_cred_a_encoded = pd.DataFrame(encoder.fit_transform(df_cred_a[["home_ownership"]]))
df_cred_a_encoded.columns = encoder.get_feature_names_out(["home_ownership"])
df_cred_a = pd.concat([df_cred_a, df_cred_a_encoded], axis=1)
df_cred_a.drop(["home_ownership"] ,axis=1, inplace=True)
df_cred_a.head()


# In[ ]:


df_cred_a["verification_status"].unique()


# In[ ]:


df_cred_a["verification_status"].value_counts()


# In[ ]:


encoder = OneHotEncoder(sparse=False)
df_cred_a_encoded = pd.DataFrame(encoder.fit_transform(df_cred_a[["verification_status"]]))
df_cred_a_encoded.columns = encoder.get_feature_names_out(["verification_status"])
df_cred_a = pd.concat([df_cred_a, df_cred_a_encoded], axis=1)
df_cred_a.drop(["verification_status"] ,axis=1, inplace=True)
df_cred_a.head()


# In[ ]:


df_cred_a['pymnt_plan'].unique()


# In[ ]:


# Define a dictionary for encoding ordinal variable
target_dict = {'n':0,
               'y':1}
# Create the mapped values in a new column
df_cred_a["pymnt_plan"] = df_cred_a["pymnt_plan"].map(target_dict)


# In[ ]:


df_cred_a.head()


# In[ ]:


df_cred_a["loan_status"].unique()


# In[ ]:


df_cred_a["purpose"].unique()


# In[ ]:


df_cred_a["purpose"].value_counts()


# * home_improvement, car, medical, wedding, moving, house, vacation, educational can be aggregated into private_use
# * renewable_energy can be aggregated into other

# In[ ]:


# Define a dictionary for aggregating variable
target_dict = {'debt_consolidation':'debt_consolidation',
               'credit_card':'credit_card',
               'home_improvement':'private_use',
               'other':'other',
               'major_purchase':'major_purchase',
               'small_business':'small_business',
               'car':'private_use',
               'medical':'private_use',
               'wedding':'private_use',
               'moving':'private_use',
               'house':'private_use',
               'vacation':'private_use',
               'educational':'private_use',
               'renewable_energy':'other'}
# Create the mapped values in a new column
df_cred_a["purpose"] = df_cred_a["purpose"].map(target_dict)


# In[ ]:


df_cred_a["purpose"].value_counts()


# In[ ]:


encoder = OneHotEncoder(sparse=False)
df_cred_a_encoded = pd.DataFrame(encoder.fit_transform(df_cred_a[["purpose"]]))
df_cred_a_encoded.columns = encoder.get_feature_names_out(["purpose"])
df_cred_a = pd.concat([df_cred_a, df_cred_a_encoded], axis=1)
df_cred_a.drop(["purpose"] ,axis=1, inplace=True)
df_cred_a.head()


# In[ ]:


df_cred_a["initial_list_status"].unique()


# In[ ]:


encoder = OneHotEncoder(sparse=False)
df_cred_a_encoded = pd.DataFrame(encoder.fit_transform(df_cred_a[["initial_list_status"]]))
df_cred_a_encoded.columns = encoder.get_feature_names_out(["initial_list_status"])
df_cred_a = pd.concat([df_cred_a, df_cred_a_encoded], axis=1)
df_cred_a.drop(["initial_list_status"] ,axis=1, inplace=True)
df_cred_a.head()


# In[ ]:


df_cred_a.info()


# In[ ]:


df_cred_a.to_csv('df_cred_a_prep1.csv', index=False)
get_ipython().system("cp 'df_cred_a_prep1.csv' '/content/drive/MyDrive/Credit Risk Assessment'")
print('Saving data is done!')


# ## Handling Outliers

# In[ ]:


df_cred_a = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/df_cred_a_prep1.csv')
df_cred_a.head(2)


# In[ ]:


df_cred_a['delinq_2yrs'].unique()


# In[ ]:


df_cred_a['inq_last_6mths'].unique()


# In[ ]:


df_cred_a['open_acc'].unique()


# In[ ]:


df_cred_a['mths_since_last_major_derog'].unique()


# In[ ]:


len(df_cred_a['mths_since_last_major_derog'].unique())


# In[ ]:


df_cred_a['pub_rec'].unique() # this is not NUMERICAL FEATURE, because it HAS RANGE


# In[ ]:


df_cred_a['collections_12_mths_ex_med'].unique() # this is not NUMERICAL FEATURE, because it HAS RANGE


# In[ ]:


df_cred_a['acc_now_delinq'].unique() # this is not NUMERICAL FEATURE, because it HAS RANGE


# In[ ]:


def subset_by_iqr(df, column):

    whisker_width=1.5
    # Calculate Q1, Q2 and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR, including optional whiskers
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter].reset_index(drop=True)


# Severe outliers:
# * installment
# * annual_inc
# * open_acc
# * total_rec_late_fee
# * last_pymnt_amnt
# * total_rev_hi_lim
# * total_coll_amt
# * collection_recovery_fee
# * tot_cur_bal
# * credit_pull_year
# 

# In[ ]:


df_cred_a.head()


# In[ ]:


numerical = ['int_rate','installment','annual_inc','dti','delinq_2yrs','inq_last_6mths','mths_since_last_delinq','open_acc',
             'revol_util','total_rec_late_fee','collection_recovery_fee','last_pymnt_amnt','mths_since_last_major_derog','tot_coll_amt',
             'tot_cur_bal','total_rev_hi_lim','pymnt_time','credit_pull_year']

outlier = ['installment','annual_inc','open_acc','total_rec_late_fee','last_pymnt_amnt','total_rev_hi_lim',
           'tot_coll_amt','collection_recovery_fee','tot_cur_bal','pymnt_time','credit_pull_year']


# In[ ]:


# Example for whiskers = 1.5, as requested by the OP
print(f'Count of rows before removing outlier: {len(df_cred_a)}')
for i in outlier:
  df_cred_a_out = subset_by_iqr(df_cred_a, i)
print(f'Count of rows after removing outlier: {len(df_cred_a_out)}')


# ## Training Test Split

# * 70% Training + 30% Testing

# In[ ]:


# Separate features and target variables (df_train)
df_train_feat = df_cred_a_out.loc[:, df_cred_a_out.columns != "loan_status"]
df_train_target = df_cred_a_out["loan_status"]


# In[ ]:


df_train_feat.to_csv('df_train_feat.csv', index=False)
get_ipython().system("cp 'df_train_feat.csv' '/content/drive/MyDrive/Credit Risk Assessment'")

df_train_target.to_csv('df_train_target.csv', index=False)
get_ipython().system("cp 'df_train_target.csv' '/content/drive/MyDrive/Credit Risk Assessment'")
print('Saving data is done!')


# In[ ]:


from collections import Counter
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_feat, df_train_target, test_size=0.3,
                                                    random_state=42, stratify=df_train_target)
print('Class from training data df_train',Counter(y_train))

print('Class from testing data df_test',Counter(y_test))


# In[ ]:


# Distribution of training target
plt.figure(figsize=(6,6))
plt.pie(
        y_train.value_counts(),
        autopct='%.2f',
        explode=[0.1,0],
        labels=["Yes","No"],
        shadow=True,
        textprops={'fontsize': 14},
        colors=["green","yellow"],
        startangle=35)

plt.title("Proportion of Class Target",fontsize=20, fontweight='bold', pad=20)
plt.show()


# # **Exploratory Data Analysis**

# Tools:
# * Matplotlib
# * Seaborn
# * Tableau

# In[ ]:


df_cred = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/df_cred.csv')
df_cred.head(2)


# ## What are the employee titles of our borrowers?
# 
# 

# In[ ]:


from wordcloud import WordCloud
wordcloud = WordCloud().generate(' '.join(emp for emp in df_cred.emp_title))

plt.figure(figsize=(10,15))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# ## Does the employment length have an impact to good or bad loan?

# ![image.png](data:image/png;base64,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gPHIdL6hH9+XTK1RrPckGneoToGeaWseNOogIiIiD2BhMpmy+BadS6XEsHvOd5xtPZy0ddM5Jj4+nu4TludsEP+agS4D6nB93kZW6a5I+IU/+dfPbdFCM2qdY1a7/1Lgp0V0zYUrmWNjY3F0dMzpMHKM+p93+5+X+w7qf17vv0gumrZkpn0BeA8azR+le/FKDicOT63iz1Hi4C4lDrld2DLCOn+VKxMHERERyZtyz7Qlc9Xsh/+MnA7iKXfxb6Y96PaekvPc3+Nz9wdXExEREXlSnr6RBxERERERyRFKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxP3xOmc6Eg3245HUKOiY2NxdHRMafDeEQP9/o9G30XERERMZ9GHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCx6wnQ26DhiUU6HIIBf+PInejzn7/2f6PFEREREcppGHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxKHkRERERExCxPRfKQkniB/avn8fWoIQTsy+lochtbano2Z/a4bgT5vsGCQc9TOZOaZT3bEuTbLd1PI5rfKqzR6J6ytvQuDmBDnQ7tWOLbjSUf1KaOXfr2mvHuc0/FW+ixM14/xv+mDaZzreJ4r86yJhHBI2nl5oyFhQXO1XowbWfU7dLE8GUMaeWGs4UFFs5udPDdSWppLFs/rYezhQXO9T5la+ydFsOm9WTsXuPj6ZiIiIhIOtY5HcBtyZfZv2oRx8u9z+vV0xdcZP2sn4hq0BPvTsksyKn4cinXJk0ZWvIk43w380+iBQVKGbDPpG7JwiksnrqIhRH3l5V2yc+2hYuYfPCeguKV6FrwEANGnYJajfioaQF2rY0Gw3/oYjjGVydvZneXnkJhzHzzfS70+Ia5YxP5b1ZVL82j72hLxuy4wHpXa64fW8y7rwxi6e6f6Oy4jo+6r6fOjB18VtsV6+vH+OntzkzaGsrnxQOYdPoj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0F8Zda0/95s86dMhgjdTHmHgxc3u9A/sxfx5M5g47xxF1R1o2ro/Q0cAjvfTsyCIlwIzKKrpRpfhHXCj+PpzqXGefZ9MY2FSDvmmmjR/aDxj61+/u2VcYH/kDjY90Z1F0x2pWZxPUsyPTNtVAICrX2/ecdrLiLVpNx7m1BnyO3dk7hNO1DJD0flcfv5hB++WjgNgbtiKxzlE0OmS9+k79hAf0I9V/UuveahSqw7rCbDpx4KL7z8peT0mkuLwvhxf0JtuR98k6S1fwIkhISGkjOuFa9+fyW/QBv+g5SwpXS34JXwAfktiSaMBbfyDWLekF2WjcuHBUOYymo9KA0WDp2bTfYAPjiNKr3nQqoOIiIhUMJvi4uIbHEVbqaJsohfP41ivf1j8GRBnz57lyVnrLDuJq9gzeGxHzizZSmQVuCtSyOGK+f7dVyy77H1OTg4ODmUP94+z4OG/47h0BUOqwJXMV9dftaj+qlt/Va4dVH9Vr1/Eek5bMip2EQET3uC/biPpr4fHXZtLUxru31UlgoNVSVxL4sB3q0RwEBERkarJek5bMqrNaMLDLD0JK5f6M8HXf1qclBeP8cy58jpoERERkdtI5Vt5EBERERERi1B4EBERERERQxQeRERERETEEIUHERERERExROFBREREREQMUXgQERERERFDFB5ERERERMQQhQcRERERETFE4UFERERERAypfE+YtkIRQUMtPQWLycnJwcHBwcKzqLrfv4iIiEhF0sqDiIiIiIgYovAgIiIiIiKGKDyIiIiIiIghCg8iIiIiImKIwoOIiIiIiBii8CAiIiIiIoYoPIiIiIiIiCEKDyIiIiIiYojCg4iIiIiIGKInTN8C/lNWWHoKt6WQw+vKbWz3FcvKbWwRERGR25VWHkRERERExBCFBxERERERMUThQUREREREDFF4EBERERERQxQeRERERETEEIUHERERERExROFBREREREQMUXgQERERERFDFB5ERERERMQQhQcRERERETFE4UFERERERAwxW3oCl4ldRMC8n8o0NOLRGW/Q18ViM7IerbsRMcStTEM2a+ZGsiT18m7Nevcj2K/2FTtfYMeyVby9H6AGbXp34YXuLtQzF3Em5RdmhezjALZ0HNCblzs7QMYh5oRFsyvv9zF78lDCduYnXSi/+kRERETE6llVeEg/mUGX8eGM8rH0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B/RjVf/Sax6q5ClLSXz6zw9Z9t2HLBtdtr02n4yJpDi87xX9j7NxXQ3af3JFc2EUc7d688JbZo5E3kHnIfUxA30fvYsvjgG6RbGIiIhYOZvi4uIbHEVbqaJsohfP41ivf+B/rXBRgc6ePcuTs9ZZcAb2DB7bkTNLthJ5m90VKeRw+X2v7iuWXd6wPoAAwrkqB1xHYnBXgty/4ePHHYA0Dsbm09jbjVqk88NbT/B0+mtEh/bC4eIehcQETSLm6VCebQRpHz/GxBoLWDrczKaAwcSM38SUShYecnJycHBwuHnH25Tqr7r1V+XaQfVX9fpFrOe0JaNiFxEw4Q3+6zaS/hYODlbBpSkN9++67YJDxVhPgI0NNjY22PRbwIJ+pa9LH/hwfEFvmk7daWikX8IH4OFog42jD4G/PM264LLBAQoPhjKX0TzVqOR9g6dm0/0zHxxt7mK682uMrWTBQURERKqmyrnyYEUsv/Jw+6rQlYerHGfBw3/HcemK61znoL8+qX7VX1Xrr8q1g+qv6vWLVL6VB5GKkLiWxIHvXjc4iIiIiFRF1nPBtIg18RjPHA9LT0JERETEumjlQUREREREDFF4EBERERERQxQeRERERETEEIUHERERERExROFBREREREQMUXgQERERERFDFB5ERERERMQQPefhFogIGmrpKVhM+T5ps+p+ryIiIiLWSCsPIiIiIiJiiMKDiIiIiIgYovAgIiIiIiKGKDyIiIiIiIghCg8iIiIiImKIwoOIiIiIiBii8CAiIiIiIoYoPIiIiIiIiCEKDyIiIiIiYojCg4iIiIiIGGK29ARuB/5TVlh6CpVayOF1t3xM9xXLbvmYIiIiIlWdVh5ERERERMQQhQcRERERETFE4UFERERERAxReBAREREREUMUHkRERERExBCFBxERERERMUThQUREREREDFF4EBERERERQxQeRERERETEEIUHERERERExROFBREREREQMMVt6AkYU5aWw/7+b2LpjD7UGBjO6jaVnVEFadyNiiFuZhmzWzI1kSeoV/ezqM/CJdjzewolaZsg/l8l3/9lKyIHzl/dzaUXIOB9M29cTuDELsKXjgN683NkBMg4xJyyaXXklXZv17slDCduZn3ShHAsUERERkcqkEoSHVLYsWEpmlxEEPFrAZ5aeTgVyc67FjmUreHv/TTo61KLGwV28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dpD6M8F/8PkPcnNJEaF83/hhHm7VjDp2kJfyHZ+tzaXhgwBpHIzNp7G3G7VI54e3hjG3+d+JaVR2hEJi5qzANTAUB6CB61l+3JlOoYeZTV8cpsl4i5QlIiIicsvcPisPIte0ngCbkuc22PRbwIJ+pa9LH/hwfEFvmk7dCUA9z/rET38cz3olfVx9Z5D59895ufRUo1/CB+DhaIONow+BvzzNyn/eT9mF68KDocxlNE+VBooGT82m+2c+ONrcxXTn1xh7G52yJCIiIlXTbXbNQ8WzzDUPt5cKu+bhKsdZ8PDfcVy6giF/4krmqn7eq+pX/VW1/qpcO6j+ql6/iFYepOpKXEviwHf/VHAQERERqYpun2seRP4oj/HM8bD0JEREREQqD608iIiIiIiIIQoPIiIiIiJiiMKDiIiIiIgYovAgIiIiIiKGKDyIiIiIiIghutvSLRARNNTSU7CYW3O/66r7/YmIiIhUJlp5EBERERERQxQeRERERETEEIUHERERERExROFBREREREQMUXgQERERERFDFB5ERERERMQQhQcRERERETFE4UFERERERAxReBAREREREUP0hOlbwH/KCktPodIJObzulo7nvmLZLR1PRERERK6mlQcRERERETFE4UFERERERAxReBAREREREUMUHkRERERExBCFBxERERERMUThQUREREREDFF4EBERERERQxQeRERERETEEIUHERERERExROFBbjuFZxL4JngiA9u6ELD+hj1J/yGYYW1dsLexwd6lLWNWJlNYujVz21QedHfCxsYel/smsS759y05RE33xa12bZx8pxOVc2nExOARzIgpvPKDRERERG4LCg9ym0lk/jMv8F2TcXw6w//GXZPmEzjnLE+vTCSruJhTMR/Q4IOxLEkDjn/Ms7OqM21nCsXFWSSGNCd85BziABIXMTv5FWJOnybhlWRmL0osGS9zJcHpzzO1rbmcaxQRERGxDOs8ykndSNDMNRQ+MoM3+rpYejaW0bobEUPcyjRks2ZuJEtSr+5aw6k+D3VtxYPtahD90TeX97nuOLZ0HNCblzs7QMYh5oRFsyuvpEez3j15KGE785MulENh5c2D8as3lLy84aoD0Ph5lq81X/wlMLv2YPjji9mQAxzaiUPgB/RwtQOgVptxfPDYUD6LAZ/UAzQeNAlncw4Ojw2iccAhoClR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1dKTEBERERGpjOGhzWjCwyw9CRERERGRqseKLpgWERERERFrpvAgIiIiIiKGKDyIiIiIiIghCg8iIiIiImKIwoOIiIiIiBii8CAiIiIiIoYoPIiIiIiIiCEKDyIiIiIiYkjle0icFYoIGmrpKVhMTk4ODg4O19ladb8XERERkduRVh5ERERERMQQhQcRERERETFE4UFERERERAxReBAREREREUMUHkRERERExBCFBxERERERMUThQUREREREDFF4EBERERERQxQeRERERETEEIUHERERERExxGzpCdwO/KessPQUrELI4XW3ZBz3FctuyTgiIiIicmtp5UFERERERAxReBAREREREUMUHkRERERExBCFBxERERERMUThQUREREREDFF4EBERERERQxQeRERERETEEIUHERERERExROFBREREREQMUXgQq1V4JoFvgicysK0LAetv2JP0H4IZ1tYFexsb7F3aMmZlMoUAnGD91Edo62KPTem2YR8fLN2WQ9R0X5xsbHDynU5UzqURE4NHMCOmsPyKExEREamEFB7ESiUy/5kX+K7JOD6d4X/jrknzCZxzlqdXJpJVXMypmA9o8MFYlqQB534lo8lLfH7wFMXFxZw6+BHtPh3BnDggcRGzk18hoaCAhFeSmb0osWS8zJUEpz/P1Lbm8i5SREREpFKxnqOj2EUEzPvpquYmA2cxpXd9C0yogrm0ImScD6bt6wncmHXDrrW8uzJ/mDvxK1cQtLekzWxXh649vOjbuiE5G9dcbAdbOg7ozcudHSDjEHPCotmVV7KlWe+ePJSwnflJF8qtrD/Pg/GrN5S8vOGqA9D4eZavNV/8YTa79mD444vZkAM08GXkmEtd7ep0oP8jdfk4Czh2gMaDJlHfDDw2iMYBh4CmRIVF03vyECv65RARERGxDtZzfNRmNOHho8s05BG76CNOdagCwYEa9HvYjejvj9PpZl1N9RnxoInNP2XT8GKjI4NHdqJezE/M2lKNiWX7u3gypE48Y18/Cm278UoPR3ZtzAL7uxlsn8C7Vhkc/iCz+bIf5MLklby9sx1TJ1/erfDMr+yNeJupO0cSOhlI9OJY0FrSez0CX67mmNcUSJzPFo9XmeZUkQWIiIiIVA7We9rS6e/ZXuNB/OpaeiLlr067TjxwfCdLU252IF8Nj15tqL/9J7adLduexbKP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zNuLV89XeamQC4cCkfAbMxMKexCFwHMk0z1v3kxEREREio67r3hoPoTgoMIehIiIiIiI/Sk6nzAtIiIiIiJFmooHERERERGxiYoHERERERGxiYoHERERERGxiYoHERERERGxiYoHERERERGxiYoHERERERGxiYoHERERERGxyd33IXFF0OqpfQt7CP+Sm+dKSUnB2dn5DoxFRERERAqbZh5ERERERMQmKh5ERERERMQmKh5ERERERMQmKh5ERERERMQmKh5ERERERMQmKh5ERERERMQmKh5ERERERMQmKh5ERERERMQmKh5ERERERMQmKh5ERERERMQmxsIeQHHQY9zSwh6CzWYd/vaWH+OxdMm/MBIRERERudto5kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGyi4kFERERERGxiLOwB5JQa+xUffxpBXIqFMjV9GDi0Py1dDYU9rFtWukodRvVpQWv3UpQyZ/D7zxFM/f405/O0LIlnh46M7VKFKqWtnD20n4+/2ktM6s36caDVY3682sYZEg8yPSiKXelZj6nj14mHDoQzN+7KHUwsIiIiIvag6Mw8JG3mkzUO9Bz/AcHBs3h3QFV+nPcdxwp7XLesBqMGuHPgu7UMGLeUXlM3sd2tDYPr521prG/iZc8TvDvta3qM+4bXI50Y0bsu5W7Wj1sD+lSMZdgbyxi2tSx9OlbI6tDpXno7HWD+v1U4nNvJ1G5uOJUogUuT/ny63/wXDc2c2TCeBzxcKJHddubOc3mbHQqkk1MJ2s48lL0jhYg3W+NSogQurd8kIiVH05kDmBT9V+cTERERkTuh6BQPx/7AscujNHRxAAw41vZjQIvjRB4p7IHdqpPM/Ggb3x5J5TJgTj/PmuhzlHXM27JmLUeiNhzkj/QrgIVTe3ayOKMmHR1v0k/V8iTtPkKi5QqJkUdJqlwBKEkr33uIXBfPv/MSO4Ww14YS67+Ns9ZMDnxSn88HTOe3/JqeWsjgCSWZ+HM8Vmsmx795hMgBo1iRkrPROb7633IenfTM9V2HQnjn2GscyMzkwGvHeCcku6g49xUzzwxlfIsiNVEmIiIiYneKTvFQyZX49WuJS7cAYElPwVzi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)

# ## Why do our borrower take credit loan?

# ![image.png](data:image/png;base64,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)

# ## What can be concluded from good or bad loans based on their number of payments on the loan (term)?

# ![image.png](data:image/png;base64,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)

# ## How about classifying grade towards our borrower and the loan status?

# In[ ]:


plt.figure(figsize=(10,5))
grade_loan = df_cred.groupby(['grade', 'loan_status'])['id'].count().reset_index()
# plot with seaborn barplot
p = sns.barplot(data=grade_loan, x='grade', y='id', hue='loan_status',palette=['red','blue'])
plt.title("Grade and Loan Status")
legend_labels, _= p.get_legend_handles_labels()
p.legend(legend_labels,['Bad Loan', 'Good Loan'])
plt.show(p)


# # **Modelling**

# In[ ]:


df_train_feat = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/df_train_feat.csv')
df_train_feat.head(2)


# In[ ]:


df_train_target = pd.read_csv('/content/drive/MyDrive/Credit Risk Assessment/df_train_target.csv')
df_train_target['loan_status']


# In[ ]:


from collections import Counter
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_feat, df_train_target['loan_status'], test_size=0.3,
                                                    random_state=42, stratify=df_train_target)
print('Class from training data df_train',Counter(y_train))

print('Class from testing data df_test',Counter(y_test))


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE

def log_transform(x):
  return np.log(x + 1)

# Pipeline to transform the numerical features
numerical = ['int_rate','installment','annual_inc','dti','delinq_2yrs','inq_last_6mths','mths_since_last_delinq','open_acc',
             'revol_util','total_rec_late_fee','collection_recovery_fee','last_pymnt_amnt','mths_since_last_major_derog','tot_coll_amt',
             'tot_cur_bal','total_rev_hi_lim','pymnt_time','credit_pull_year']
skewed = ['installment','annual_inc','delinq_2yrs','inq_last_6mths','mths_since_last_delinq','open_acc',
          'revol_util','total_rec_late_fee','collection_recovery_fee','last_pymnt_amnt','mths_since_last_major_derog','tot_coll_amt',
          'tot_cur_bal','total_rev_hi_lim','pymnt_time','credit_pull_year']
diff = list(set(numerical) - set(skewed))

smt = SMOTE(random_state=42)
ss = StandardScaler()
log_transformer = FunctionTransformer(log_transform) # remainder='passthrough'

numerical_transformer = Pipeline([('log', log_transformer),('ss', ss)])
ct = ColumnTransformer([('num_transformer', numerical_transformer, skewed), ('scaler', ss, diff)], remainder='passthrough')


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


# In[ ]:


def evaluate_ks_and_roc_auc(y_real, y_proba):
    # Unite both visions to be able to filter
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba[:, 1]

    # Recover each class
    class0 = df[df['real'] == 0]
    class1 = df[df['real'] == 1]

    ks = ks_2samp(class0['proba'], class1['proba'])
    roc_auc = roc_auc_score(df['real'] , df['proba'])

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"KS: {ks.statistic:.4f} (p-value: {ks.pvalue:.3e})")
    return ks.statistic, roc_auc


# In[ ]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[ ]:


# Main pipeline for fitting.
model_LR = Pipeline([
                   ('column_transformer', ct),
                   ('smt', smt),
                   ('RF', LogisticRegression(random_state=42) )
          ])
model_LR.fit(X_train, y_train)
print("Training is success!")
y_pred = model_LR.predict_proba(X_test)
predicted = model_LR.predict(X_test)
#print AUC, KS score, and classification report
ks, auc = evaluate_ks_and_roc_auc(y_test, y_pred)
matrix = classification_report(y_test, predicted)
print('Classification report Logistic Regression : \n',matrix)
cm = confusion_matrix(y_test, predicted)
target_names = ["Bad Loan","Good Loan"]
plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None,normalize=False)


# In[ ]:


len(model_LR.named_steps['RF'].coef_[0])


# ## Random Forest

# In[ ]:


# Main pipeline for fitting.
model_RF = Pipeline([
                   ('column_transformer', ct),
                   ('smt', smt),
                   ('RF', RandomForestClassifier(random_state=42) )
          ])
model_RF.fit(X_train, y_train)
print("Training is success!")
y_pred = model_RF.predict_proba(X_test)
predicted = model_RF.predict(X_test)
#print AUC, KS score, and classification report
ks, auc = evaluate_ks_and_roc_auc(y_test, y_pred)
matrix = classification_report(y_test, predicted)
print('Classification report Random Forest Classifier : \n',matrix)
cm = confusion_matrix(y_test, predicted)
target_names = ["Bad Loan","Good Loan"]
plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None,normalize=False)


# ## Gradient Boosting Classifier

# In[ ]:


# Main pipeline for fitting.
model_GB = Pipeline([
                   ('column_transformer', ct),
                   ('smt', smt),
                   ('GB', GradientBoostingClassifier(random_state=42) )
          ])
model_GB.fit(X_train, y_train)
print("Training is success!")
y_pred = model_GB.predict_proba(X_test)
predicted = model_GB.predict(X_test)
#print AUC, KS score, and classification report
ks, auc = evaluate_ks_and_roc_auc(y_test, y_pred)
matrix = classification_report(y_test, predicted)
print('Classification report Gradient Boosting Classifier : \n',matrix)
cm = confusion_matrix(y_test, predicted)
target_names = ["Bad Loan","Good Loan"]
plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None,normalize=False)


# ## XGBoost Classifier

# In[ ]:


# Main pipeline for fitting.
model_XGB = Pipeline([
                   ('column_transformer', ct),
                   ('smt', smt),
                   ('XGB', xgb.XGBClassifier(objective="binary:logistic",random_state=42) )
          ])
model_XGB.fit(X_train, y_train)
print("Training is success!")
y_pred = model_XGB.predict_proba(X_test)
predicted = model_XGB.predict(X_test)
#print AUC, KS score, and classification report
ks, auc = evaluate_ks_and_roc_auc(y_test, y_pred)
matrix = classification_report(y_test, predicted)
print('Classification report XGBoost Classifier : \n',matrix)
cm = confusion_matrix(y_test, predicted)
target_names = ["Bad Loan","Good Loan"]
plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None,normalize=False)


# ## Voting Classifier

# In[ ]:


from sklearn.ensemble import VotingClassifier

clf1 = RandomForestClassifier(random_state=42)
clf2 = GradientBoostingClassifier(random_state=42)
clf3 = xgb.XGBClassifier(objective="binary:logistic",random_state=42)

# Main pipeline for fitting.
model_VC = Pipeline([
                   ('column_transformer', ct),
                   ('smt', smt),
                   ('VC', VotingClassifier(estimators=[('RF', clf1), ('GB', clf2), ('XGB', clf3)],
                        voting='soft', weights=[1,2,1]) )
          ])
model_VC.fit(X_train, y_train)
print("Training is success!")
y_pred = model_VC.predict_proba(X_test)
predicted = model_VC.predict(X_test)
#print AUC, KS score, and classification report
ks, auc = evaluate_ks_and_roc_auc(y_test, y_pred)
matrix = classification_report(y_test, predicted)
print('Classification report Voting Classifier : \n',matrix)
cm = confusion_matrix(y_test, predicted)
target_names = ["Bad Loan","Good Loan"]
plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None,normalize=False)


# # **Model Optimization & Evaluation**

# Chosen model: XGBoost Classifier

# ## Hyperparameter Tuning
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define a range of hyperparameters
param_grid = {
    'XGB__n_estimators': [100, 200, 300],
    'XGB__max_depth': [3, 4, 5],
    'XGB__learning_rate': [0.01, 0.1, 0.2],
    # Add other parameters here
}

# Create a GridSearchCV object
grid_search = GridSearchCV(model_XGB, param_grid, scoring='roc_auc', cv=3, verbose=2)


# ## Fit the Model with Grid Search

# In[ ]:


grid_search.fit(X_train, y_train)


# ## Best Parameters and Model

# In[ ]:


print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_


# ## Model Evaluation

# In[ ]:


y_pred_proba = best_model.predict_proba(X_test)
print(y_pred_proba.shape)

predicted = best_model.predict(X_test)
# Evaluate AUC and KS Score
ks, auc = evaluate_ks_and_roc_auc(y_test, y_pred_proba)

# Classification Report and Confusion Matrix
matrix = classification_report(y_test, predicted)
print('Classification report for Optimized XGBoost Classifier:\n', matrix)

cm = confusion_matrix(y_test, predicted)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=False)



# In[ ]:


print(f'AUC Score: {auc}')
print(f'KS Score: {ks}')


# ## Model Checking

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

scores_train = []
scores_test = []

xgb_model = model_XGB
kf = KFold(shuffle=True, random_state=42, n_splits=5)

for train_index, test_index in kf.split(df_train_feat):
    # Separate training and testing data based on index
    X_train, X_test = df_train_feat.iloc[train_index], df_train_feat.iloc[test_index]
    y_train, y_test = df_train_target.iloc[train_index], df_train_target.iloc[test_index]

    # Train the model with training data
    xgb_model.fit(X_train, y_train)

    # Make predictions on training and testing data
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)

    # Calculating the mean absolute error for training and testing
    score_train = mean_absolute_error(y_train, y_train_pred)
    score_test = mean_absolute_error(y_test, y_test_pred)

    scores_train.append(score_train)
    scores_test.append(score_test)

# Visualization of K-Fold Cross-Validation results
folds = range(1, kf.get_n_splits() + 1)
plt.plot(folds, scores_train, 'o-', color='green', label='Training MAE')
plt.plot(folds, scores_test, 'o-', color='red', label='Testing MAE')
plt.legend()
plt.grid()
plt.xlabel('Fold Number')
plt.ylabel('Mean Absolute Error')
plt.title('K-Fold Validation for XGBoost Model')
plt.show()


# # **Model Interpretation**

# Top 10 Feature Importances in the Model

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Extract the features and importance of the XGBoost model
feature_importances = pd.DataFrame()
feature_importances['feature'] = X_train.columns
feature_importances['importance'] = model_XGB.named_steps['XGB'].feature_importances_

# Visualization of the 10 most important features
plt.figure(figsize=(10, 6))
plot = feature_importances.sort_values('importance', ascending=False).head(10).plot.barh(color='blue', legend=None)
plot.set_yticklabels(feature_importances.sort_values('importance', ascending=False).head(10).feature)
plt.title('10 Most Important Features in the XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()


# Top 3 feature importances in predicting credit risk is good or bad:
# 
# * `delinq_2yrs`: The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
# * `inq_last_6mths`: Number of credit inquiries in past 12 months
# * `total_rec_late_fee`: Late fees received to date
