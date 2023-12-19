import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from prettytable import PrettyTable
import warnings
warnings.simplefilter(action='ignore')
pd.set_option('display.max_columns', 100)


def reverse_standardize_data(df, y):
    original_df = (df * y.std()) + y.mean()
    # print('Mean(original):',y.mean())
    # print('STD:',y.std())
    return original_df


# Read Data
df = pd.read_csv('Debt_Regression_cleaned.csv')
# print(df.head())
print('Shape:', df.shape)
# print(df.columns)
# Drop additional columns
# df.drop(['Unnamed: 0'], axis=1, inplace=True)
y_org = df['Target_Original']
print('Y-Original Mean:', np.mean(y_org))
df.drop('Target_Original', axis=1, inplace=True)
X, y = df.drop('Outstanding_Debt', axis=1), df['Outstanding_Debt']
print('Shape X:', X.shape)
# print(X.columns)
# X = X.head(40000)
# y = y.head(40000)
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

# Result Table:
result_table = PrettyTable(field_names=['Model', 'R-squared', 'AIC', 'BIC', 'Adj_R2','MSE'])
result_table.title = 'Multiple Linear Regression Result'
# Regression Analysis using OLS
# We have list of selected and eliminated features
print('Stepwise Regression Analysis (OLS)')

OLS_features = list(X_train.columns)
OLS_features_eliminated = []
table = PrettyTable(field_names=['Drop Feature','AIC','BIC','AdjR2','p-value','f-statistic'])
best_adj_R2 = 0
first = 0
while len(OLS_features) > 0:
    # print('Entered')
    X_train = sm.add_constant(X_train[OLS_features])
    model = sm.OLS(y_train, X_train).fit()
    if first == 0:
        print('Model Summary before Regression Analysis OLS')
        print(model.summary())
        first += 1
    print('F-value:', round(model.fvalue,3))
    curr_adj_R2 = model.rsquared_adj
    max_p_value = model.pvalues.max()
    print('Max p-value:', round(max_p_value,3))
    print('Current Adjusted R-squared:', round(curr_adj_R2,3))
    max_p_value_index = model.pvalues.idxmax()
    p_values = model.pvalues[0:]
    threshold = 0.01
    if max_p_value > threshold:
        # print('Entered if')
        # if curr_adj_R2 >= best_adj_R2:
        drop_feature = max_p_value_index
        OLS_features_eliminated.append(drop_feature)
        OLS_features.remove(drop_feature)
        print('Feature to be dropped:', drop_feature)
        table.add_row([drop_feature,round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(max_p_value, 3),
                       round(model.fvalue,3)])
        # table = table.append({'Drop Feature': drop_feature, 'AIC': model.aic,
        #                       'BIC': model.bic, 'AdjR2': model.rsquared_adj,
        #                       'p-value': round(max_p_value, 3), 'f-statistic': round(model.fvalue,3),
        #                       }, ignore_index=True)
        best_adj_R2 = model.rsquared_adj
        # else:
        #     print('Adj-Rsquared saturation point achieved')
        #     print(model.summary())
        #     break
    else:
        print('Regression Analysis complete')
        print(model.summary())
        break
print(table)
# table.to_csv('Regression_Analysis_OLS.csv')
print('Final Adj R-squared:', round(model.rsquared_adj, 3))
print('Final F-statistic:',round(model.fvalue, 3))
# Drop features based on correlation matrix
# OLS_features.remove('Credit_History_Age')
# OLS_features.remove('Monthly_Inhand_Salary')
print('Eliminated Features:', OLS_features_eliminated)
print('Selected features:', OLS_features)
print(f'After Regression {len(OLS_features)} features are selected')

# Update Result table
# result_table.add_row(['Multiple Linear Regression',mlr_model.])
# Create Train and Test Sets based on selected features
X_train = X_train[OLS_features]
X_test = X_test[OLS_features]
# mlr_OLS = sm.OLS(sm.add_constant(X_train),y_train).fit()
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred_reg = mlr_model.predict(X_test)
# y_pred_reg = mlr_OLS.predict(sm.add_constant(X_test))
# Our predictions are standardized, so we will reverse standardize them based on y_original
y_pred_org = reverse_standardize_data(y_pred_reg, y_org)
y_test_org = reverse_standardize_data(y_test, y_org)
y_train_org = reverse_standardize_data(y_train, y_org)
print('y-train shape:',y_train_org.shape)
print('y-test shape:',y_test_org.shape)
print('y-pred shape:',y_pred_org.shape)

# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred_reg)
# mae = metrics.mean_absolute_error(y_test, y_pred_reg)
print('Mean Squared Error:', mse.round(3))
# print('Root Mean Squared Error:', round(np.sqrt(mse), 3))
# print('Mean Absolute Error:', round(mae, 3))
# Update Result table
result_table.add_row(['Multiple Linear Regression', round(model.rsquared, 3), round(model.aic,3),
                      round(model.bic, 3), round(model.rsquared_adj, 3), round(mse, 3)])
print('Results of Linear Regression')
print(result_table)

# ----------------------------------------------------------------
#       Random Forest Selected Features Linear Regression
# ----------------------------------------------------------------
print('--------Random Forest feature selection based Linear Regression--------')
df_rf = pd.read_csv('rf_regression.csv')
rf_cols = df_rf.columns
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(df_rf.drop('Outstanding_Debt',axis=1),
                                                                df_rf['Outstanding_Debt'], random_state=5805)
X_train_rf = sm.add_constant(X_train_rf)
model_rf_ols = sm.OLS(y_train, X_train).fit()
y_pred_rf = model_rf_ols.predict(X_test)
y_pred_rf_org = reverse_standardize_data(y_pred_rf, y_org)
y_train_rf_org = reverse_standardize_data(y_train_rf, y_org)
# MSE
mse_rf = metrics.mean_squared_error(y_test, y_pred_rf)
# mae_rf = metrics.mean_absolute_error(y_test, y_pred_rf)
result_table.add_row(['Multiple Linear Regression(RForest)', round(model_rf_ols.rsquared, 3), round(model_rf_ols.aic,3),
                      round(model_rf_ols.bic, 3), round(model_rf_ols.rsquared_adj, 3), round(mse_rf, 3)])

# ---------------------------------------------------
#       Evaluation of Linear Regression Model
# ----------------------------------------------------
print('Evaluation of Linear Regression Model')
print(result_table)
# plot_result_df = pd.DataFrame({'Original test set': y_test_org, 'Predicted Debt': y_pred_org,
#                                'Training Debt': y_train_org})
# x_range = [i for i in range(len(plot_result_df['Original test set']))]
train_range = [i for i in range(len(y_train_org))]
test_range = [i for i in range(y_test.shape[0])]
plt.plot(train_range, y_train_org, label='Training Debt')
plt.plot(test_range, y_test_org, label='Test Debt')
plt.plot(test_range, y_pred_org, label='Predicted Debt')
plt.xlabel('Observations')
plt.ylabel('Debt')
plt.legend(loc='best')
plt.title('Linear Regression (OLS Analysis)')
plt.show()


# ----------------------------------------------------
#               Prediction Intervals
# ----------------------------------------------------
# X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=5805,shuffle=True)
# X_test = X_test[OLS_features]
sm_pred = model.get_prediction(sm.add_constant(X_test)) \
          .summary_frame(alpha=0.05)
print('--------------------Prediction Intervals--------------------')
print(sm_pred.head())
# obs_ci_lower and obs_ci_upper are prediction intervals
lower_interval = sm_pred['obs_ci_lower']
upper_interval = sm_pred['obs_ci_upper']
y_test_original = reverse_standardize_data(sm_pred['mean'], y_org)
lower_interval_original = reverse_standardize_data(lower_interval, y_org)
upper_interval_original = reverse_standardize_data(upper_interval, y_org)
#print('Upper Interval:',upper_interval_original.head())
x_range = [i for i in range(len(y_test_original))]
plt.plot(x_range, lower_interval_original, alpha=0.4, label='Lower interval')
plt.plot(x_range, upper_interval_original, alpha=0.4, label='Upper interval')
plt.plot(x_range, y_test_original, alpha=1.0, label='Predicted Debt')
plt.title('Predicted Debt with Intervals')
plt.ylabel('Debt')
plt.xlabel('Observations')
plt.legend()
plt.show()
print('--------Confidence Interval Analysis------------')
print('Confidence Intervals:')
print(model.conf_int())




