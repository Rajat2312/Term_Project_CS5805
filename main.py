# ----------------------------------------------------------------
#           Term Project - CS 5805 Machine Learning 1
# ----------------------------------------------------------------
# Author: Rajat Belgundi

# -----------------Project is conducted in four phases----------------------
# Phase 1: Feature Engineering and Exploratory Data Analysis
# Phase 2: Regression Analysis [on selected continuous numerical feature]
# Phase 3: Classification Analysis
# Phase 4: Association Rule Mining and Clustering
# ----------------------------------------------------------------------------

# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, recall_score, precision_score,classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time
pd.set_option('display.max_columns', 100)

# import phase1  # Phase1
# Basic EDA
def basic_eda(data):
    # Display first 5 rows
    print(data.head())
    # Inspect numerical columns
    print(data.describe())
    # Inspect memory usage
    print(data.info(memory_usage='deep'))
    print('Data Description')
    print(data.describe().T)
    print('Shape of Data')
    print('Rows: {}'.format(data.shape[0]))
    print('Columns: {}'.format(data.shape[1]))


def missing_value_check(data):
    # Check missing values
    # print(self.data.isna().sum())
    missing_data = pd.DataFrame(data.isna().sum())
    print(missing_data)


def imbalance_check(data, col):
    print(data[col].value_counts(normalize=True))  # Count in %
    sns.countplot(x=data[col])
    plt.title('Imbalance Check for '+col)
    plt.show()


def clean_object_data(data):
    # Strip the additional underscores
    char_remove = ['!@9#%8', '_']
    for char in char_remove:
        data = data.replace(char, "")
    return data


def clean_numeric_data(data):
    try:
        return float(data.replace("_", ""))
    except:
        return np.nan


def credit_history_age(age):
    # 15 Years and 11 Months is the format
    if pd.notnull(age):
        age_str = age.split()
        return round(int(age_str[0]) * 12 + int(age_str[3]),2)


def standardize_data(df):
    standardized_df = (df - df.mean(numeric_only=True)) / (df.std(numeric_only=True))
    return standardized_df


def conduct_rforest_selection(X, y, kind):
    if kind == 'C':
        print('------------------METHOD 1: RANDOM_FOREST CLASSIFICATION------------------')
        clf_model = RandomForestClassifier(n_estimators=100, random_state=5805)
        clf_model.fit(X, y)
        # Save model as pickle
        # model_file = 'rf_model.pkl'
        # with open(model_file, 'wb') as file:
        #     pickle.dump(model, file)

        # Load from pickle
        # with open(model_file, 'rb') as file:
        #     model = pickle.load(file)
        importances = clf_model.feature_importances_
        # Check Feature Importance
        feature_importance = pd.Series(clf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(30, 16))
        sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
        plt.title('RForest Classifier based Feature Importance')
        plt.show()
        selected_features = []
        eliminated_features = []
        threshold = 0.016
        # feature_importance series has the mapping of features and importances
        # Iterate through series using items function to get index and value
        for feature, importance in feature_importance.items():
            if importance > threshold:
                selected_features.append(feature)
            else:
                eliminated_features.append(feature)
        print('No. of features selected by RForest Analysis:', len(selected_features))
        print('(RForest Classifier Analysis) Selected Features with threshold = {}:'.format(threshold))
        print(selected_features)
        print('(RForest Classifier Analysis) Eliminated Features with threshold = {}:'.format(threshold),
              eliminated_features)
        print('----------------------------------------------------------------------------------------------')
        return selected_features

    if kind == 'R':
        print('------------------METHOD 1: RANDOM_FOREST REGRESSION------------------')
        reg_model = RandomForestRegressor(n_estimators=100, random_state=5805)
        reg_model.fit(X_reg, y_reg)
        importances = reg_model.feature_importances_
        # Check Feature Importance
        feature_importance = pd.Series(reg_model.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
        plt.figure(figsize=(30, 16))
        sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
        plt.title('RForest Regressor based Feature Importance')
        plt.show()
        selected_features = []
        eliminated_features = []
        threshold = 0.01
        # feature_importance series has the mapping of features and importances
        # Iterate through series using items function to get index and value
        for feature, importance in feature_importance.items():
            if importance > threshold:
                selected_features.append(feature)
            else:
                eliminated_features.append(feature)
        print('No. of features selected by RForest Regressor Analysis:', len(selected_features))
        print('(RForest Regressor Analysis) Selected Features with threshold = {}:'.format(threshold))
        print(selected_features)
        print('(RForest Regressor Analysis) Eliminated Features with threshold = {}:'.format(threshold),eliminated_features)
        print('----------------------------------------------------------------------------------------------')
        return selected_features


def conduct_PCA(X, y, kind):
    if kind == 'C':
        print('-----------------------------------------------------------------------')
        print('          Principal Component Analysis (PCA) for Classification        ')
        print('-----------------------------------------------------------------------')
        pca = PCA(random_state=5805)
        pca.fit(X, y)
        print('Shape of X: ', X.shape)
        print('Columns of X:', X.columns)
        # print('Explained variance:',pca.explained_variance_)
        # print('Cumulative sum of explained variance ratio',np.cumsum(pca.explained_variance_ratio_))
        n_comp = [i for i in range(1, X.shape[1] + 1)]
        cum_exp_variance = np.cumsum(pca.explained_variance_ratio_)
        print('Explained Variance:', cum_exp_variance)
        plt.figure(figsize=(18, 10))
        print('N-comp:', len(n_comp))
        print('Exp:', len(cum_exp_variance))
        plt.plot(n_comp, np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
        plt.axhline(y=0.9, linestyle='-', color='red')
        # print(x=)
        plt.axvline(x=len([var for var in cum_exp_variance if var < 0.9]) + 1, linestyle='-', color='red')
        # plt.plot(y=0.9)
        plt.xticks(n_comp)  # x-ticks start from 1
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative Sum of Explained Variance Ratio')
        plt.title('Selecting # of components for PCA - Classification Dataset')
        plt.show()
        return pca

    if kind == 'R':
        print('--------------------------------------------------------------------')
        print('          Principal Component Analysis (PCA) for Regression         ')
        print('--------------------------------------------------------------------')
        pca_reg = PCA(random_state=5805)
        pca_reg.fit(X,y)
        print('Shape of X: ', X.shape)
        print('Columns of X:', X.columns)
        # print('Explained variance:',pca.explained_variance_)
        # print('Cumulative sum of explained variance ratio',np.cumsum(pca.explained_variance_ratio_))
        n_comp = [i for i in range(1, X.shape[1] + 1)]
        cum_exp_variance = np.cumsum(pca_reg.explained_variance_ratio_)
        print('Explained Variance:', cum_exp_variance)
        plt.figure(figsize=(18, 10))
        print('N-comp:', len(n_comp))
        print('Exp:', len(cum_exp_variance))
        plt.plot(n_comp, np.cumsum(pca_reg.explained_variance_ratio_), marker='o', linestyle='-')
        plt.axhline(y=0.9, linestyle='-', color='red')
        # print(x=)
        plt.axvline(x=len([var for var in cum_exp_variance if var < 0.9]) + 1, linestyle='-', color='red')
        # plt.plot(y=0.9)
        plt.xticks(n_comp)  # x-ticks start from 1
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative Sum of Explained Variance Ratio')
        plt.title('Selecting # of components for PCA - Regression Dataset')
        plt.show()
        return pca_reg


def get_condition_num(X, y, pca, n):
    print(' ------------------------------ ')
    print('        Condition Number        ')
    print(' ------------------------------ ')
    pca_selected = PCA(n_components=n, random_state=5805)
    pca_selected.fit(X,y)
    cov_matrix = pca.get_covariance()
    cov_selected = pca_selected.get_covariance()
    # Condition number
    condition_num = np.linalg.cond(cov_matrix)
    c_num_selected = np.linalg.cond(cov_selected)
    print('Condition Number (pre PCA):', condition_num)
    print('Condition Number (post PCA) and {} components:'.format(n), c_num_selected)


def truncated_svd(X, kind):
    if kind == 'C':
        print('--------------------------------------------------------------------')
        print('          Singular Value Decomposition (SVD) for Classification     ')
        print('--------------------------------------------------------------------')
        svd = TruncatedSVD(n_components=(X.shape[1]-1))
        svd.fit(X)

        # Original data shape
        print("Original Data Shape:", X.shape)

        # Reduced data shape
        # print("Reduced Data Shape:", data_reduced.shape)
        n_comp = [i for i in range(1, X.shape[1])]
        # Explained variance ratio
        print("Explained Variance Ratio:", svd.explained_variance_ratio_.sum())

        # Visualize explained variance
        cum_exp_variance = np.cumsum(svd.explained_variance_ratio_)
        plt.figure(figsize=(14, 6))
        plt.plot(n_comp, np.cumsum(svd.explained_variance_ratio_))
        plt.axhline(y=0.9, linestyle='-', color='red')
        # print(x=)
        plt.axvline(x=len([var for var in cum_exp_variance if var < 0.9])+1, linestyle='-', color='red')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('SVD Analysis for Classification')
        plt.xticks(n_comp)
        # plt.grid()
        plt.show()
    if kind == 'R':
        print('--------------------------------------------------------------------')
        print('          Singular Value Decomposition (SVD) for Regression         ')
        print('--------------------------------------------------------------------')
        svd = TruncatedSVD(n_components=(X.shape[1] - 1))
        svd.fit(X)

        # Original data shape
        print("Original Data Shape:", X.shape)

        # Reduced data shape
        # print("Reduced Data Shape:", data_reduced.shape)
        n_comp = [i for i in range(1, X.shape[1])]
        # Explained variance ratio
        print("Explained Variance Ratio:", svd.explained_variance_ratio_.sum())

        # Visualize explained variance
        cum_exp_variance = np.cumsum(svd.explained_variance_ratio_)
        plt.figure(figsize=(14, 6))
        plt.plot(n_comp, np.cumsum(svd.explained_variance_ratio_))
        plt.axhline(y=0.9, linestyle='-', color='red')
        # print(x=)
        plt.axvline(x=len([var for var in cum_exp_variance if var < 0.9]), linestyle='-', color='red')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('SVD Analysis for Regression')
        plt.xticks(n_comp)
        # plt.grid()
        plt.show()


# compute the vif for all given features
def compute_vif(considered_features):
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    #     X = add_constant(X)

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']
    return vif


if __name__ == "__main__":
    start = time.time()
    # Read dataset as Pandas DataFrame
    df = pd.read_csv('credit_score_classification_data.csv', low_memory=False)

    # ------------------------------OBSERVATIONS--------------------------------
    # 1. We will inspect the features (numeric and categorical)
    # 2. We will check for missing values
    # 3. We will drop columns like ID, Name, SSN (Contain NaN values and no impact on the classification task)
    # 4. Dataset has no duplicates - as checked below
    # 5. We see that Customer_ID has 12500 unique values that occur 8 times each. (Groupby to fill missing values)
    # Candidates for Regression - Outstanding Debt, Monthly Balance
    # ---------------------------------------------------------------------------
    # Basic EDA
    basic_eda(df)

    # Missing Value Analysis
    print('Before dropping general info columns')
    missing_value_check(df)
    print('After dropping general info columns')

    # Drop general info columns by passing list of columns to drop
    df.drop(['Name', 'ID', 'SSN'],axis=1,inplace=True)
    missing_value_check(df)
    print(df.columns)
    # -------------------------------------------------
    #                  Drop NaN values
    # -------------------------------------------------
    # df.dropna(how='any',inplace=True,axis=0)
    # print('Dropped NaN values')
    # Check duplicates
    print('-----------Check Duplicates-------------')
    df.duplicated().value_counts()
    print('No duplicates found')
    print('----------------------------------------')

    # Check for imbalance for Target feature Credit_Score
    imbalance_check(df, 'Credit_Score')
    print('--------------------Encoding Target Feature--------------------')
    df['Credit_Score'].replace({'Good': 2, 'Standard': 1, 'Poor': 0}, inplace=True)

    # Check if Customer_ID is unique
    # We will use this column to fill missing values
    # print(df['Customer_ID'].value_counts())

    # -------------------------------------------------------
    #             Data Cleaning - Numerical columns
    # -------------------------------------------------------
    print(' -------------------------------------------------------')
    print('             Data Cleaning - Numerical columns          ')
    print(' -------------------------------------------------------')

    num_cols_to_fix = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                       'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']
    for num_col in num_cols_to_fix:
        print('Column Name: ' + num_col)
        print("**" * 24)
        print(df[num_col].value_counts())
        print('END', "--" * 22, '\n')
    # We wil replace if any string like '_' with empty string but with np.nan
    for i in num_cols_to_fix:
        df[i] = df[i].apply(clean_numeric_data)
    print('------------After Cleaning Numerical columns of object type----------------')
    for num_col in num_cols_to_fix:
        print('Column Name: ' + num_col)
        print("**" * 24)
        print(df[num_col].value_counts(dropna=False))
        print('END', "--" * 22, '\n')
    # print(df[df['Amount_invested_monthly']])
    print(df.info())

    # # Check unique values in each numerical column
    # # numeric = df.select_dtypes(exclude='object')
    # print('Numerical features described:')
    # print(df_cleaned.describe(exclude='object').T)
    # print('Categorical features described:')
    # print(df_cleaned.describe(exclude=np.number).T)
    # print('Missing Value Check')
    # missing_value_check(df_cleaned)

    # We are now left with the following object type columns
    # --------------------------------------------------------
    #         Fix Strange values in Object type columns
    # --------------------------------------------------------
    print('--------------------------------------------------------')
    print('         Data Cleaning in Object type columns      ')
    print('--------------------------------------------------------')
    # Fix strange values in Object type columns
    object_cols = df.select_dtypes(include='object').columns
    print(object_cols)
    for col in object_cols:
        print('Column Name: ' + col)
        print("**" * 24)
        print(df[col].value_counts(dropna=False))
        print('END', "--" * 22, '\n')

    # ['Month','Occupation','Type_of_Loan','Credit_Mix','Credit_History_Age','Payment_of_Min_Amount']
    # -------------------------------------------------------------
    #         Replace strange values in the object type columns
    # -------------------------------------------------------------
    print('-------------------------------------------------------------')
    print('        Replace strange values in the object type columns    ')
    print('------------------------------------------------------------')
    # 1. Month
    # print(df['Month'].value_counts(dropna=False))
    print(df['Month'].dtypes)
    # month_tab = df.groupby('Credit_Score')['Month'].value_counts()

    # 2. Occupation
    # Replace garbage value(_______) with mode based on customer_id
    print('-------------------------Occupation-------------------------')
    print('Before')
    print(df['Occupation'].value_counts())
    # Replace with mode w.r.t Customer_ID
    df['Occupation'] = df['Occupation'].replace('_______', np.nan)
    mode_fn = lambda x: x.mode().iat[0]
    df['Occupation'] = df['Occupation'].fillna(df.groupby('Customer_ID')['Occupation'].transform(mode_fn))
    print('After')
    print(df['Occupation'].value_counts())
    plt.figure(figsize=(24, 12))
    sns.countplot(data=df, x='Occupation', hue='Credit_Score')
    plt.title('Occupation wise Credit Score')
    plt.show()

    # # 3. Type_of_Loan
    print('-------------------------Type of Loan-------------------------')
    print('Before')
    print(df['Type_of_Loan'].value_counts().head(10))
    # # We see that we have comma separated values which we can convert to columns
    # # We can look at the Types of Loans as categorical columns
    # # We have types - Credit Builder Loan, Personal Loan, Mortgage Loan, Student Loan
    # # We will now assign 0 or 1 to each column of unique loan types
    # # We also have a category Not Specified which we can either include or drop
    # # Now replacing Type of Loan could have a great effect on credit score hence replacing would be counterproductive
    # # Hence we include only types other than Not Specified and we split them up into columns
    # # tl = ''
    # # print(type('kadka'))
    # # for i in df['Type_of_Loan'].values:
    # #
    # #     if type(i) == str:
    # #         tl += i + ','
    # # print(len(tl))
    # # uniqueLoanTypes = set(tl.split(","))
    # # print(uniqueLoanTypes)
    #
    uloantypes = ['Debt Consolidation Loan', 'Student Loan', 'Mortgage Loan', 'Auto Loan',
                  'Personal Loan', 'Not Specified', 'Payday Loan', 'Student Loan', 'Mortgage Loan',
                  'Home Equity Loan', 'Debt Consolidation Loan', 'Student Loan', 'Credit-Builder Loan',
                  'Payday Loan', 'Auto Loan', 'Mortgage Loan', 'Personal Loan', 'Auto Loan', 'Debt Consolidation Loan',
                  'Credit-Builder Loan', 'Home Equity Loan', 'Not Specified', 'Not Specified', 'Credit-Builder Loan',
                  'Payday Loan', 'Personal Loan', 'Home Equity Loan']
    unique_loan_types = set(uloantypes)
    # print('We have {} unique loan types: ',len(unique_loan_types))
    # print(unique_loan_types)
    # Handle Type of Loan =  Not Specified
    # df['Type_of_Loan'].dropna(how='any', axis=0, inplace=True)
    df.dropna(how='any', axis=0, subset=['Type_of_Loan'], inplace=True)
    # print(df.groupby('Type_of_Loan')['Customer_ID'].value_counts(dropna=False))
    # print(df[df['Type_of_Loan'].isna()]['Customer_ID'].head())
    # One-Hot Encoding of Type_of_Loan column
    print('One-Hot Encoding of Type_of_Loan column')
    for i in unique_loan_types:
        typeLoan = []
        for value in df['Type_of_Loan']:
            if type(value) == str:
                typeLoan.append(1 if i in value else 0)
        df[i] = typeLoan
    print('After handling Type of Loan column:', df.columns)
    print(df.tail(10))
    # We can now drop the column Type_of_Loan column
    df.drop('Type_of_Loan', axis=1, inplace=True)
    # Avoid Dummy variable trap
    df.drop('Not Specified',axis=1,inplace=True)
    # # df['Type_of_Loan'].value_counts()
    # # for i in df['Type_of_Loan'].values:
    # #     flag = 0
    # #     if(type(i) == str):
    # #         for typeLoan in uloantypes:
    # #             if typeLoan in i:
    # #                 flag = 1
    # #         if flag == 0:
    # #             print(i)

    # Working
    # df = pd.concat([df, df['Type_of_Loan'].str.split(', ', expand=True)], axis=1)
    # print('After handling Type of Loan column:', df.columns)
    # print(df.tail())

    # 4. Credit_Mix
    print('------------------- Credit_Mix -------------------')
    print('Before Replace')
    print(df['Credit_Mix'].value_counts())
    # Replace with mode
    df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
    mode_fn = lambda x: x.mode().iat[0]
    df['Credit_Mix'] = df['Credit_Mix'].fillna(df.groupby('Customer_ID')['Credit_Mix'].transform(mode_fn))
    # df['Credit_Mix'].replace({'_': 'Standard'}, inplace=True)  # replace _ with mode = Standard
    print('After replace')
    print(df['Credit_Mix'].value_counts())

    # # 5. Credit_History_Age
    print('------------------- Credit_History_Age -------------------')
    # # It is in the form of years and months we will convert to numeric values in the form of Months
    # # We have NaN values as well here
    # # We will replace with mean of Months
    # # 15 Years and 11 Months is the format
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(credit_history_age)
    print(df['Credit_History_Age'].head(10))
    # Fill NaN values
    mean_fn = lambda x: x.mean()
    df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(mean_fn)
    print('Misisng Value Check after filling NaN values')
    print('There are {} missing values after cleaning:'.format(df['Credit_History_Age'].isna().sum()))
    print(df['Credit_History_Age'].head(10))

    # Payment_of_Min_Amount
    print('------------------- Payment_of_Min_Amount -------------------')
    # If payment of min amount due has been done or not
    print(df['Payment_of_Min_Amount'].value_counts(dropna=False))
    print(df['Payment_of_Min_Amount'].isna().sum())
    mode_fn = lambda x: x.mode().iat[0]
    df['Payment_of_Min_Amount'].replace('NM', np.nan, inplace=True)
    df['Payment_of_Min_Amount'] = df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(mode_fn)
    print(df['Payment_of_Min_Amount'].value_counts(dropna=False))
    sns.countplot(data=df, x='Payment_of_Min_Amount', hue='Credit_Score')
    plt.title('Min Amount Due based Credit Score')
    plt.show()

    # Check df
    print(df.shape)
    missing_value_check(df)
    print(df.info())

    # ---------------------------------------------------------
    #               Handle remaining NaN values
    # ---------------------------------------------------------

    # 1. Monthly_Inhand_Salary
    # We can replace with mean of Customer_ID
    df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].transform(mean_fn)

    # 2. Num_of_Delayed_Payment
    # We replace with mean of Customer_ID
    df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(mean_fn)

    # 3. Changed_Credit_Limit
    # Replace with mean of Customer_ID
    df['Changed_Credit_Limit'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(mean_fn)

    # 4. Num_Credit_Inquiries
    df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(mean_fn)

    # 5. Amount_invested_monthly
    df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(mean_fn)

    # 6. Monthly_Balance
    df['Monthly_Balance'] = df.groupby('Customer_ID')['Monthly_Balance'].transform(mean_fn)

    # Missing value check
    print('After cleaning remaining NaN values:')
    missing_value_check(df)

    # Check Numerical columns for wrong values
    # Example: Age in negative, Monthly_Balance in negative
    # use describe to check
    # we can ignore the encoded columns like types of loan
    numeric_df = df.select_dtypes(exclude='object')
    print(numeric_df.describe().T.head(17))

    # Observations
    # 1. Age: Min value = -500, Max value = 867  clean up age
    # 2. Num_Bank_Accounts: min value = -1 we can make it 0
    # 3. Num_Credit_Card: Outlier Detection
    # 4. Interest_Rate:  Outlier Detection
    # 5. Num_of_Loan: Min value = -100
    # 6. Delay_from_due_date: Min value = -5
    # 7. Num_of_Delayed_Payment: Min value = -1
    # 8. Changed_Credit_Limit: Min value = -1.07 (Explore Later)
    # 9. Num_Credit_Inquiries: Min value = 0.0  Max value: 600
    # 10. Outstanding_Debt: Min Value = 0.23  Max value = 4998
    # 11. Credit_Utilization_Ratio: Min value = 20  Max value = 49.5
    # 12. Credit_History_Age: Min value = 4.5  Max value = 400
    # 13. Total_EMI_per_month: Min value = 4.46  Max value = 82331
    # 14. Amount_invested_monthly: Min value = 15.2  Max value = 1313
    # 15. Monthly_Balance: Min value = large -ve  Max value = 1313, garbage value = 3333333333
    # 16. Annual_Income: Min value = 7005   Max value = large +ve
    # 17. Monthly_Inhand_Salary: Min value = 303.6  Max value = 15,204

    # --------------------------------------------------
    #                  OUTLIER DETECTION
    # --------------------------------------------------

    # 1. Age
    print('--------------------------------------------')
    print('                  Age                       ')
    print('--------------------------------------------')
    # Handle negative values and values greater than 100
    df = df[(df['Age'] > 0) & (df['Age'] < 100)]
    print(df.shape)
    imbalance_check(df,'Credit_Score')

    # 2. Num_Bank_Accounts
    # Replace -1 with 0
    df.loc[df['Num_Bank_Accounts'] < 0, 'Num_Bank_Accounts'] = 0
    sns.boxplot(data=df, x='Num_Credit_Card')
    plt.title('Number of Bank Accounts with Outliers')
    plt.show()
    # print('Outlier instances in Num_Bank_Accounts: ', df[df['Num_Bank_Accounts'] > 8 & df['Num_Bank_Accounts']].shape[0])
    q25 = df['Num_Bank_Accounts'].quantile(0.25)
    q75 = df['Num_Bank_Accounts'].quantile(0.75)
    iqr = q75 - q25
    upper = q75 + 1.5 * iqr
    # lower = q25 - 1.5 * iqr
    lower = 0
    df = df[(df['Num_Bank_Accounts'] >= lower) & (df['Num_Bank_Accounts'] <= upper)]
    print('After Removing Outliers from # of Bank Accounts')
    print(df.shape)

    # 3. Num_Credit_Card
    print('--------------------------------------------')
    print('              Num_Credit_Card               ')
    print('--------------------------------------------')
    # Detect Outliers we can use IQR range to detect outliers
    sns.boxplot(data=df, x='Num_Credit_Card')
    plt.title('Number of Credit Cards with Outliers')
    plt.show()
    print('Outlier instances in Num_Credit_Card: ',df[df['Num_Credit_Card']>8].shape[0])
    q25 = df['Num_Credit_Card'].quantile(0.25)
    q75 = df['Num_Credit_Card'].quantile(0.75)
    iqr = q75 - q25
    upper = q75 + 1.5 * iqr
    # lower = q25 - 1.5 * iqr
    lower = 0
    df = df[(df['Num_Credit_Card'] >= lower) & (df['Num_Credit_Card'] <= upper)]
    print('After Removing Outliers from # of Credit Cards')
    print(df.shape)
    sns.boxplot(data=df, x='Interest_Rate')
    plt.title('Interest Rate after Outlier Removal')
    plt.show()

    # Interest_Rate
    # Detect Outliers after checking distribution
    print('--------------------------------------------')
    print('              Interest_Rate                 ')
    print('--------------------------------------------')
    sns.boxplot(data=df, x='Interest_Rate')
    plt.title('Interest Rate before Outlier Removal')
    plt.show()
    q25 = df['Interest_Rate'].quantile(0.25)
    q75 = df['Interest_Rate'].quantile(0.75)
    iqr = q75 - q25
    print('25th',q25)
    print('75th',q75)
    print('IQR of Interest Rate',iqr)
    upper = q75 + 1.5 * iqr
    # lower = q25 - 1.5 * iqr
    lower = 0.0  # Interest Rate low is 0.0
    print('Interest Rate IQR')
    print('Upper:', upper)
    print('Lower:', lower)
    print('After Removing Outliers from Interest_Rate')
    df = df[(df['Interest_Rate'] >= lower) & (df['Interest_Rate'] <= upper)]
    print(df.shape)
    sns.boxplot(data=df, x='Interest_Rate')
    plt.title('Interest Rate after Outlier Removal')
    plt.show()

    # Num_of_Loan
    print('--------------------------------------------')
    print('                Num_of_Loan                 ')
    print('--------------------------------------------')
    # We have -ve values which is not right, we will set lower limit to 0
    df.loc[df['Num_of_Loan']<0, 'Num_of_Loan'] = 0
    sns.boxplot(data=df, x='Num_of_Loan')
    plt.title('Num_of_Loan before Outlier Removal')
    plt.xlim(min(df['Num_of_Loan']), max(df['Num_of_Loan']))
    plt.title('Number of Loans before Outlier Removal')
    plt.show()
    q25 = df['Num_of_Loan'].quantile(0.25)
    q75 = df['Num_of_Loan'].quantile(0.75)
    iqr = q75 - q25
    print('25th:', q25)
    print('75th:', q75)
    print('IQR:', iqr)
    upper = q75 + 1.5 * iqr
    # lower = q25 - 1.5 * iqr
    lower = 0.0
    print('# of Loans IQR')
    print('Upper:', upper)
    print('Lower:', lower)
    print('After Removing Outliers from # of Loans')
    df = df[(df['Num_of_Loan'] >= lower) & (df['Num_of_Loan'] <= upper)]
    df['Num_of_Loan'] = df['Num_of_Loan'].astype(int)
    # Check credit score of very high Num_of_Loan
    sns.boxplot(data=df,x='Num_of_Loan',hue='Credit_Score')
    plt.title('Num_of_Loan after outlier removal')
    plt.show()
    print(df[(df['Num_of_Loan'] >= lower) & (df['Num_of_Loan'] <= upper)].shape)
    print('Credit Score based on Num of Loans!!!')
    # loans_mean = df.groupby('Credit_Score')['Num_of_Loan'].mean()
    # num_loan_tab = pd.pivot_table(data=df, values=['Credit_Score','Num_of_Loan'], index='Credit_Score')
    # pd.crosstab(index=df['Credit_Score'],columns=loans_mean))
    print(df[['Num_of_Loan', 'Credit_Score']].groupby('Credit_Score').mean())

    # sns.countplot(data=df,x='Num_of_Loan',hue='Credit_Score')

    # Delay_from_due_date
    print('--------------------------------------------')
    print('            Delay_from_due_date             ')
    print('--------------------------------------------')
    sns.boxplot(data=df, x='Delay_from_due_date')
    plt.title('Delay_from_due_date before outlier removal')
    plt.show()
    print('No. of negative values:', df[df['Delay_from_due_date'] < 0].shape[0])
    print('Filtering out negative values from Delay_from_due_date')
    print('After removing outliers from Delay_from_due_date')
    df = df[df['Delay_from_due_date'] >= 0]
    sns.boxplot(data=df, x='Delay_from_due_date')
    plt.title('Delay_from_due_date after outlier removal')
    plt.show()
    # print(df[df['Delay_from_due_date'] < 0])
    print('No.of records:', df.shape[0])
    print('-----Delay_from_due_date vs Credit Score-----')
    print(df[['Delay_from_due_date', 'Credit_Score']].groupby('Credit_Score').mean())

    # Num_of_Delayed_Payment
    print('--------------------------------------------')
    print('           Num_of_Delayed_Payment           ')
    print('--------------------------------------------')
    print('Min:', min(df['Num_of_Delayed_Payment']))
    print('Max:', max(df['Num_of_Delayed_Payment']))
    q25 = df['Num_of_Delayed_Payment'].quantile(0.25)
    q75 = df['Num_of_Delayed_Payment'].quantile(0.75)
    iqr = q75 - q25
    print('IQR:',iqr)
    upper = q75 + 1.5 * iqr
    print("Upper:", upper)
    print('-----Num_of_Delayed_Payment vs Credit Score-----')
    print(df[['Num_of_Delayed_Payment', 'Credit_Score']].groupby('Credit_Score').mean())

    # Changed_Credit_Limit
    print('--------------------------------------------')
    print('            Changed_Credit_Limit            ')
    print('--------------------------------------------')
    print(df['Changed_Credit_Limit'].value_counts(dropna=False))
    print('Min:',min(df['Changed_Credit_Limit']))
    print('Max:',max(df['Changed_Credit_Limit']))
    # We are capturing the change so keep it above zero
    q25 = df['Changed_Credit_Limit'].quantile(0.25)
    q75 = df['Changed_Credit_Limit'].quantile(0.75)
    iqr = q75 - q25
    print('IQR:', iqr)
    upper = q75 + 1.5 * iqr
    print("Upper:", upper)
    print('-----Monthly Balance vs Credit Score-----')
    print(df[['Monthly_Balance', 'Credit_Score']].groupby('Credit_Score').mean())

    # Monthly_Balance
    print('--------------------------------------------')
    print('             Monthly_Balance                ')
    print('--------------------------------------------')
    print('# of Negative values are:', df[df['Monthly_Balance'] < 0].shape[0])
    # Include only positive values
    df = df[df['Monthly_Balance'] > 0]
    # We are capturing the change so keep it above zero
    q25 = df['Monthly_Balance'].quantile(0.25)
    q75 = df['Monthly_Balance'].quantile(0.75)
    iqr = q75 - q25
    print('IQR:', iqr)
    upper = q75 + 1.5 * iqr
    print("Upper:", upper)
    print('We will include only positive(>0) values')
    print('-----Monthly Balance vs Credit Score-----')
    # print(df[df['Monthly_Balance']==-333333333333333333333333333].sample(2))
    print(df[['Monthly_Balance', 'Credit_Score']].groupby('Credit_Score').mean())

    # # Outstanding_Debt
    print('--------------------------------------------')
    print('             Outstanding_Debt                ')
    print('--------------------------------------------')
    q25 = df['Outstanding_Debt'].quantile(0.25)
    q75 = df['Outstanding_Debt'].quantile(0.75)
    iqr = q75 - q25
    print('IQR:', iqr)
    upper = q75 + 1.5 * iqr
    print("Upper:", upper)

    # Basic EDA
    # basic_eda(df)

    # ----------------------Split into X & y---------------------
    X, y = df.drop('Credit_Score', axis=1), df['Credit_Score']
    reg_target_org = df['Outstanding_Debt'].head(50000).values

    assoc_df = X[['Occupation','Credit_Mix','Payment_of_Min_Amount','Debt Consolidation Loan','Auto Loan',
                 'Credit-Builder Loan','Home Equity Loan',
                  'Personal Loan','Student Loan']].sample(10000,random_state=5805)
    print(assoc_df.columns)
    # assoc_df.to_csv('Association.csv',index=False)
    print('Association Rule Mining Dataset is ready!')
    # -------------------------------------------------------------
    #                       Standardization
    # -------------------------------------------------------------

    df_standardized = standardize_data(X)

    # ------------------------------------------------------------------
    #                   One-Hot Encoding
    # ------------------------------------------------------------------
    print('------------------------------------------------------')
    print('                  One-Hot Encoding                    ')
    print('------------------------------------------------------')
    object_df = df.select_dtypes(include='object')
    # print(object_df.head())
    # ['Payment_of_Min_Amount', 'Credit_Mix', 'Occupation', 'Payment_Behaviour','Month']
    print('-----Encoding in progress-----')
    one_hot_df = pd.get_dummies(data=object_df[['Payment_of_Min_Amount', 'Credit_Mix', 'Occupation', 'Payment_Behaviour', 'Month']], drop_first=True)
    # Payment_of_Min_Amount: {'No':0, 'Yes':1}
    # print(one_hot_df.head())
    print('Columns:', one_hot_df.columns)
    # Concat one_hot_df with original df
    df = pd.concat([df_standardized, one_hot_df], axis=1)
    df = pd.concat([df, y], axis=1)
    df.drop(['Payment_of_Min_Amount', 'Credit_Mix', 'Occupation', 'Payment_Behaviour', 'Month'], axis=1, inplace=True)

    # Now that we have used Customer_ID as reference for filling our missing values and data cleaning
    # We can now drop Customer_ID
    df.drop('Customer_ID', axis=1, inplace=True)
    # basic_eda(df)
    # print(df.head())
    print('Shape of data before X-y split:',df.shape)

    X, y = df.drop('Credit_Score', axis=1), df['Credit_Score']
    # --------------------------------------------------
    #             Prepare Regression Dataset
    # --------------------------------------------------
    X_reg,y_reg = df.drop('Outstanding_Debt', axis=1), df['Outstanding_Debt']
    # --------------------------------------------------------
    #                   Feature Importance
    # ---------------------------------------------------------
    # print('---------------------------------------------------------------')
    # print('                 Feature Selection Techniques                  ')
    print('---------------------------------------------------------------')
    print('Random Forest based feature selection for Classification Dataset')
    clf_selected_features = conduct_rforest_selection(X, y, kind='C')
    print('Random Forest based feature selection for Regression Dataset')
    reg_selected_features = conduct_rforest_selection(X_reg, y_reg, kind='R')
    rf_reg = X_reg[reg_selected_features]
    rf_reg = rf_reg.head(50000)

    # # df_reg = X_reg[reg_selected_features]
    df_reg = X_reg
    df_reg = df_reg.head(50000)
    # df_reg = df_reg.head(50000)
    # df_target = pd.DataFrame(reg_target_org)
    df_target = pd.DataFrame(y_reg)
    rf_reg = pd.concat([rf_reg, df_target.head(50000)], axis=1)
    print('Original Regression Dependent Mean:', np.mean(reg_target_org))
    df_target = df_target.head(50000)
    df_reg = pd.concat([df_reg, df_target], axis=1)
    df_reg['Target_Original'] = reg_target_org
    print('Regression Data shape (after feature selection):', df_reg.shape)
    # rf_reg.to_csv('rf_regression.csv', index=False)
    print('Regression Data Random Forest Features dataset ready')
    # df_reg.to_csv('Debt_Regression_cleaned.csv',index=False)
    print('Regression Data csv file ready!')

    # ------------------------------------------------------------------------
    #           Principal Component Analysis (PCA) and Condition Number
    # ------------------------------------------------------------------------
    print('METHOD 2: Principal Component Analysis (PCA) for Regression')
    X_reg = X_reg.head(50000)
    y_reg = y_reg.head(50000)
    pca_reg = conduct_PCA(X_reg, y_reg, kind='R')
    print('Condition Number for Regression Dataset')
    get_condition_num(X_reg, y_reg, pca_reg, n=25)
    print('METHOD 2: Principal Component Analysis (PCA) for Classification')
    pca_clf = conduct_PCA(X, y, kind='C')
    print('Condition Number for Classification Dataset')
    get_condition_num(X, y, pca_clf, n=25)

    # ---------------------------------------------------
    #          Singular Value Decomposition(SVD)
    # ---------------------------------------------------
    # print('SVD reading is in process')
    print('METHOD 3: Singular Value Decomposition (SVD) for Classification')
    truncated_svd(X, kind='C')
    print('METHOD 3: Singular Value Decomposition (SVD) for Regression')
    truncated_svd(X_reg, kind='R')

    ## ---------------------------------------------------
    ##              Variance Inflation Factor(VIF)
    ## ---------------------------------------------------
    print('METHOD 4: VIF (Variance Inflation Factor)')
    features = list(X_reg.columns)
    df_vif = compute_vif(features)
    print(df_vif.head(10).sort_values(by='VIF',ascending=False))

    # ----------------------------------------------------
    #              Balancing the dataset
    # ----------------------------------------------------
    # Check imbalance
    imbalance_check(df, col='Credit_Score')
    # We can see that we have imbalanced dataset, we will use SMOTE oversampling to balance the dataset
    over_sampling = SMOTE(sampling_strategy='auto', random_state=5805)
    X_columns = X.columns
    X, y = over_sampling.fit_resample(X, y)
    print('Oversampled X:', X.shape)
    down_sample = RandomUnderSampler(sampling_strategy={0: 18000, 1: 18000, 2: 18000}, random_state=5805)
    X, y = down_sample.fit_resample(X, y)
    enn_down_sample = TomekLinks(sampling_strategy='all', n_jobs=3)
    X, y = enn_down_sample.fit_resample(X, y)
    df1 = pd.DataFrame(X, columns=list(X_columns))
    df2 = pd.DataFrame(y,columns=['Credit_Score'])
    df3 = pd.concat([df1, df2], axis=1)
    # Check the balance
    print('Balance the dataset')
    imbalance_check(df3, col='Credit_Score')
    # print(df3.columns)
    # selected_features = ['Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date', 'Credit_Utilization_Ratio', 'Changed_Credit_Limit', 'Credit_Mix_Standard', 'Num_Credit_Inquiries', 'Credit_History_Age', 'Num_of_Delayed_Payment', 'Credit_Mix_Good', 'Num_Credit_Card', 'Total_EMI_per_month', 'Age', 'Monthly_Balance', 'Annual_Income', 'Amount_invested_monthly', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_of_Loan', 'Payment_of_Min_Amount_Yes', 'Month_July', 'Month_August', 'Payment_Behaviour_Low_spent_Small_value_payments', 'Month_March', 'Month_February', 'Month_January']
    print('Shape of cleaned df:', df3[clf_selected_features].shape)
    df3.drop_duplicates(inplace=True)
    print('Shape of cleaned df:', df3[clf_selected_features].shape)
    # df4 = pd.concat([df3[clf_selected_features],df2])
    # #Save preprocessed and balanced dataset into new csv for Classification dataset
    clf_selected_features.append('Credit_Score')
    # df3[clf_selected_features].to_csv('credit_classification_cleaned.csv', index=False)
    print('Classification Data csv file ready!')

    # # ----------------------------------------------------
    # #               Correlation Analysis
    # # ----------------------------------------------------
    print('------------------------------------------------')
    print('            Correlation Analysis                 ')
    print('------------------------------------------------')
    temp_df = df.drop(list(one_hot_df.columns), axis=1)
    loan_list = []
    for i in uloantypes:
        if i in list(temp_df.columns):
            loan_list.append(i)
    temp_df.drop(loan_list,axis=1,inplace=True)
    corr = temp_df.corr()
    # # print(temp_df.columns)
    plt.figure(figsize=(24,18))
    plt.title('Correlation Matrix')
    sns.heatmap(corr,annot=True)
    plt.show()

    # # ----------------------------------------------------
    # #               Covariance Analysis
    # # ----------------------------------------------------
    print('------------------------------------------------')
    print('            Covariance Analysis                 ')
    print('------------------------------------------------')
    # temp_df = df.drop(list(one_hot_df.columns),axis=1)
    cov = temp_df.cov()
    plt.figure(figsize=(24, 18))
    plt.title('Covariance Matrix')
    sns.heatmap(cov, annot=True)
    plt.show()

    end = time.time()
    print('Time taken in minutes:', round((end-start)/60,2))




