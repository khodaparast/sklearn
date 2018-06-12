import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('D:/Kaggle/House_Price/train.csv')
test = pd.read_csv('D:/Kaggle/House_Price/test.csv')

def plotPivot(xName):
    pivot = train.pivot_table(index=xName, values='SalePrice', aggfunc=np.median)
    pivot.plot(kind='bar', color='blue')
    plt.xlabel(xName)
    plt.ylabel('Median SalePrice')
    plt.xticks(rotation=0)
    plt.show()



# print(train.shape)# prints number of rows and columns
# print(test.shape)# prints number of rows and columns

# print(train.head(n=10)) # prints 10 top row of trian
# print(train.SalePrice.describe())

# plt.hist(train.SalePrice,color='blue')# it plots histogram of sale price that how much is skew
# plt.show()

target = np.log(train.SalePrice)
# print(target)
# plt.hist(target,color='blue') # it plots histogram of sale price after log that is less skewed
# plt.show()


numeric_features = train.select_dtypes(include=[np.number])  # selects columns with number type
# print(numeric_feature.dtypes)


corr = numeric_features.corr()  # compute correlation between SalePrice and other features
# print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n') # show top 5 positive correlated
# print(corr['SalePrice'].sort_values(ascending=False)[-5:]) # show top 5 negative correlated

# print(train.OverallQual.unique()) # it print unique values of the feature

quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)  # create pivot table
# print(quality_pivot)

# quality_pivot.plot(kind='bar', color='blue')# plot pivot tabel computed in previous step   PERFECT!!!!!!!!!!!!!
# plt.xlabel('Overall Quallity')
# plt.ylabel('Median Sale Price')
# plt.xticks(rotation=0)
# plt.show()


# plt.scatter(x=train['GrLivArea'], y=target) # plot relation between GrLivArea and Price
# plt.ylabel(' Sale Price ')
# plt.xlabel('living area square feet')
# plt.show()



# plt.scatter(x=train['GarageArea'], y=target) # plot relation between Garage Area and Price
# plt.ylabel(' Sale Price ')
# plt.xlabel('Garage Area')
# plt.show()

# Remove Outliers
# train = train[train['GarageArea'] < 1200] # remove outlier
# train = train[train['GarageArea'] > 0] # remove outlier
#
# plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))  # plot relation between Garage Area and Price after outlier removal
# plt.xlim(-200,1600) # This forces the same scale as before
# plt.ylabel(' Sale Price ')
# plt.xlabel('Garage Area')
# plt.show()


# -----------------------------------------Handling Null values-------------------------------

nulls=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25]) # list the null values of a data frame
nulls.columns=['Null Count']
nulls.index.name='Feature'
# print(nulls)
#
# print(" Unique vallues are: ",train.MiscFeature.unique())# unique values of feature MiscFeature

categoricals=train.select_dtypes(exclude=[np.number]) # One-hot encoding is a technique which will transform categorical data into numbers so the model can understand whether or not a particular observation falls into one category or another.
# print(categoricals.describe())

# ---------------------------------------------feature engineering ---> feature engineering must apply in train and test
# print(train.Street.value_counts(),"\n")

# print(train.Street.unique())
# print(train.Street.describe())
def encode(x): return 1 if x == 'Pave' else 0
train['enc_Street'] = train.Street.apply(encode)
test['enc_Street'] = test.Street.apply(encode)
#
train=train.drop('Street', 1)
test=test.drop('Street', 1)
# print(train)

# train=pd.concat([train, pd.get_dummies(train.Street)], axis=1)# change attribute Street to numeric, add to table, remove street
# train=train.drop('Street', 1)
# print(train)
# test=pd.concat([test, pd.get_dummies(test.Street)], axis=1)# change attribute Street to numeric, add to table, remove street
# test=test.drop('Street', 1)# in this command 1=column for drop row can set 0
# print(test)

# enc_street_train=pd.get_dummies(train.Street )
# enc_street_test=pd.get_dummies(test.Street)
# print(enc_street_train.describe(),'\n \n')
# print(train.Street.describe())

# plotPivot('SaleCondition')
#Notice that Partial has a significantly higher Median Sale Price than the others. We will encode this as a new feature. We select all of the houses where SaleCondition is equal to Patrial and assign the value 1, otherwise assign 0.
def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

train=train.drop('SaleCondition', 1)
test=test.drop('SaleCondition', 1)
# plotPivot('SaleCondition')


# feature engineering SaleType
# print(train.SaleType.unique())
# plotPivot('SaleType')

def encode(x): return 1 if x == 'New'  else 0
train['enc_saleType'] = train.SaleType.apply(encode)
test['enc_saleType'] = test.SaleType.apply(encode)

train=train.drop('SaleType', 1)
test=test.drop('SaleType', 1)
# print(train)

# print(train.Utilities.unique())
# plotPivot('Utilities')

def encode(x): return 1 if x == 'AllPub'  else 0
train['enc_Utilities'] = train.Utilities.apply(encode)
test['enc_Utilities'] = test.Utilities.apply(encode)

train=train.drop('Utilities', 1)
test=test.drop('Utilities', 1)
# print(train)
# plotPivot('Utilities')

# print(train.LotConfig.unique())
# plotPivot('LotConfig')

def encode(x):
     if x == 'FR2':
         return 1
     elif x==('FR3' or 'CulDSac'):
         return 2
     else:
         return 3

train['enc_LotConfig'] = train.LotConfig.apply(encode)
test['enc_LotConfig'] = test.LotConfig.apply(encode)
train=train.drop('LotConfig', 1)
test=test.drop('LotConfig', 1)
# print(train)
# plotPivot('LotConfig')



# print(train.Neighborhood.unique())
# plotPivot('Neighborhood')

def encode(x):
    if x == ('MeadowV' or 'IDOTRR' ):
        return 1
    elif x == ('Blueste' or 'BrkSide' or 'Edwards' or 'NAmes' or 'OldTown' or 'SWISU' or 'SawyerW'):
        return 2
    elif x== ('Blmngtn' or 'CollgCr' or 'ClearCr' or 'Crawfor' or 'Gilbert' or 'NWAmes' or 'SawyerW' ):
        return 3
    else:
        return 2

train['enc_Neighborhood'] = train.Neighborhood.apply(encode)
test['enc_Neighborhood'] = test.Neighborhood.apply(encode)
train = train.drop('Neighborhood', 1)
test = test.drop('Neighborhood', 1)
# plotPivot('enc_Neighborhood')



# plotPivot('LotShape')
# print(train.LotShape.unique())

def encode(x):
    if x == 'IR1':
        return 1
    elif x == 'IR2':
        return 2
    elif x== 'IR3':
        return 3
    else:
        return 4

train['enc_LotShape'] = train.LotShape.apply(encode)
test['enc_LotShape'] = test.LotShape.apply(encode)
train = train.drop('LotShape', 1)
test = test.drop('LotShape', 1)
# plotPivot('LotShape')



# print(train.Condition1.unique())
# plotPivot('Condition1')

def encode(x):
    if x == ('Artery' or 'Feedr' or 'RRAe'):
        return 1
    elif x == ('Norm' or 'RRAe' ):
        return 3
    elif x==( 'RRNn' or 'PosN' ):
        return 4
    else:
        return 2


train['enc_Condition1'] = train.Condition1.apply(encode)
test['enc_Condition1'] = test.Condition1.apply(encode)
train = train.drop('Condition1', 1)
test = test.drop('Condition1', 1)
plotPivot('enc_Condition1')


# print(train.LandSlope.unique())
# plotPivot('LandSlope')

def encode(x):
    if x == 'Gtl':
        return 1
    elif x == 'Sev':
        return 2
    else:
        return 3

train['enc_LandSlope'] = train.LandSlope.apply(encode)
test['enc_LandSlope'] = test.LandSlope.apply(encode)
train = train.drop('LandSlope', 1)
test = test.drop('LandSlope', 1)
# plotPivot('enc_LandSlope')


# print(train.LandContour.unique())
# plotPivot('LandContour')

def encode(x):
    if x == 'Bnk':
        return 1
    elif x == 'Lvl':
        return 2
    elif x== 'Low':
        return 3
    else:
        return 4

train['enc_LandContour'] = train.LandContour.apply(encode)
test['enc_LandContour'] = test.LandContour.apply(encode)
train = train.drop('LandContour', 1)
test = test.drop('LandContour', 1)
# plotPivot('enc_LandContour')



# plotPivot('Alley')
train = train.drop('Alley', 1)
test = test.drop('Alley', 1)



# plotPivot('LotFrontage')
data = train.select_dtypes(include=[np.number]).interpolate().dropna() # (Interpolation) for missing value in LotFrontage
sum(data.isnull().sum() != 0)
# print(data)


# print(train)
# print(train.LotFrontage.unique())
# print(train.MSZoning.unique())

#--------------------------------- model building--------------------------
Y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
# print(X)
print(X.shape)
print(Y.shape)

# from sklearn.model_selection import train_test_split  #new version  --> pip install -U scikit-learn
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=.33)

from sklearn import linear_model
lr = linear_model.LinearRegression()
# lr=linear_model.ARDRegression()

model = lr.fit(X_train, y_train)

print ("Model Score : \n", model.score(X_test, y_test)) # evaluate model

predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
# print (' RMSE is:  \n', mean_squared_error(y_test, predictions))


actual_values = y_test
# plt.scatter(predictions, actual_values, alpha=.75,
#             color='b') #alpha helps to show overlapping data
# plt.xlabel('Predicted Price')
# plt.ylabel('Actual Price')
# plt.title('Linear Regression Model')
# plt.show()


#---------- tune model by selecting different value for alpha --------------------------
for i in range (-2, 3): # change alpha to fit model
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay, xy=(12.1, 10.6), size='x-large')
    # plt.show()

# --------------------------------------------submission----------------------------------------------------------------

submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(
    include=[np.number]).drop(['Id'], axis=1).interpolate()


predictions = model.predict(feats)
final_predictions = np.exp(predictions)

# print ("Original predictions are: \n", predictions[:5], "\n")
# print ("Final predictions are: \n", final_predictions[:5])

submission['SalePrice'] = final_predictions
submission.head()

# submission.to_csv('D:/submission.csv', index=False) # 2454 in Kaggle
# ----------------------------------------------------------------------------------------------------------------------
