
##################################### QUESTION - 01 #############################################
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# reading csv data
d1 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/Machine_Learning/Salaries-Simple_Linear.csv')
print(d1)

d1.shape #to get dimension of dataset

# report number of variables
print(d1.columns)

#missing values for each variables
d1.isnull().sum()

X = d1[["Years_of_Expertise"]]
Y = d1[["Salary"]]
X = sm.add_constant(X)

#Simple linear regression model
model = sm.OLS(Y, X).fit()
model.summary()

X = d1[["Years_of_Expertise"]]
sns.set(color_codes=True)
sns.regplot(X, Y)




######################################### QUESTION - 02 ##################################################

d2 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/Machine_Learning/3-Products-Multiple.csv')
print(d2)

d2.shape #to get dimension of dataset
d2.dtypes
# report number of variables
print(d2.columns)

#missing values for each variables
d2.isnull().sum()
d3 = d2[['Product_1','Product_2','Product_3','Profit']]

#correlation matrix
corr = d3.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

#Scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(d3, figsize=(6, 6), diagonal='kde',color = 'r')


mod = smf.ols("Profit ~ Product_1+Product_2+Product_3+C(Location)", data=d2).fit()
mod.summary()
mod.params

def forward_selected(data, response):

    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

model = forward_selected(d2, 'Profit')
print(model.model.formula)
print(model.rsquared_adj)
model.summary()

# RESULT:
# Our final regression, equation is: Profit = 4.698e^04 + (7.966e^-01)Product_1 + (2.991e^-02)Product_3.
# Since our reference location is city_1, based on the regresson analysis, investing more in
# city_1 on product_1 and product_3 would yeild better profit. Product_1 yeilds more profit than product_3 and
# we can conclude this from beta coefficients.

############################ QUESTION - 03 #########################################


d4 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/Machine_Learning/Propose-Salaries-Polynomial.csv')
print(d4)

d4.shape #to get dimension of dataset
d4.dtypes
# report number of variables
print(d4.columns)

d4['Level2'] = np.power(d4[['Level']],2)
X = d4[['Level','Level2']]
X = sm.add_constant(X)
Y = d4[['Salary']]

#Scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(d4, figsize=(4, 4), diagonal='kde',color = 'r')


lm = sm.OLS(Y, X).fit()
lm.summary()
lm.params

####Predict value for Level = 6.5
x0 = pd.DataFrame([{"Const":1,"Level": 6.5, "Level2": np.power(6.5, 2)}])
lm.predict(x0)