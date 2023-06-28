# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:13:02 2023

@author: domingosdeeulariadumba
"""


# Informing the directory of the file 

%pwd



""" Importing the required libraries """


# Libraries/modules for EDA and plotting

import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as mpl
mpl.style.use('ggplot')


# ML library/modules

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lreg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as SScl, normalize


# Library/module to analyse statistical parameters

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


# Library to save, load the model and make predictions

import joblib


# Library to ignore warnings about compactibility issues and so on

import warnings
warnings.filterwarnings('ignore')



"""" EXPLORATORY DATA ANALYSIS """


# Loading the dataset

df_re= pd.read_csv("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/realestate_dataset.csv")


# Setting the option to show all the columns in the dataset

pd.set_option('display.max_colwidth', None)


# Displaying the first and last ten records

df_re.head(10)

df_re.tail(10)


# Checking the data type

df_re.dtypes


# Looking to the columns in the dataset

df_re.columns


# Dropping the 'No' column from the dataset, since it does not hold any significance
# to our analysis. 

df_re= df_re.drop(['No'], axis=1)


# Statisctical summary of the variables

df_re.describe()


# Looking for possible null values and NaN

df_re.isnull().value_counts()

df_re.isna().value_counts()


# Distribution plots

variables = ['X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
       'Y house price of unit area']
for k in variables:
    sb.displot(df_re[k])
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/displot1.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/displot2.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/displot3.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/displot4.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/displot5.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/displot6.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/displot7.png")
mpl.close()


# Displaying the KDE plots

for i in variables:
    mpl.figure()
    sb.kdeplot(df_re[i])
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/kdeplot1.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/kdeplot2.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/kdeplot3.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/kdeplot4.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/kdeplot5.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/kdeplot6.png")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/kdeplot7.png")
mpl.close()    

    
# Visualizing the pair plot

sb.pairplot(df_re)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/initialbarplot.png")
mpl.close()


# Correlation heatmap

sb.heatmap(df_re.corr(), annot=True, cmap='coolwarm')
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/correlation_heatmap.png")
mpl.close()


# Analysing the different combination between the target and each independent
# variable

sb.pairplot(df_re,x_vars=['X2 house age',
           'X3 distance to the nearest MRT station',
           'X4 number of convenience stores'], y_vars=['Y house price of unit area'],
                height=5, aspect=.8, kind="reg")
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/finalpairplot.png")
mpl.close()
 
    """
    Since it's more visualizable, next we dispaly the strip plot for Convenience
    stores.
    """    
       
sb.stripplot(x='X4 number of convenience stores', y='Y house price of unit area', data=df_re)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/stripplot_conveniencestores.png")
mpl.close()



""" DATA PREPROCESSING """

# Setting up the independent and dependent variables

X=pd.DataFrame(df_re.iloc[:,:-1])

y=pd.DataFrame(df_re.iloc[:,-1])


# Normalizing, scaling and reducing the dimensionality of the preditors set

    """
    The next function automatizes the process of mormalizing, escaling and
    reduction of dimensionality
    """
    
def Normalize_Scale_and_Reduce(X):
    
    
    # Normalizing the data since there's different ranges among the predictors
    
    norm_pred=normalize(X)
    
    
    # Scaling the normalized data to make them comparable
    
    scl_pred=SScl().fit_transform(norm_pred)
    
    
    # Reducing the dimensionality of the scaled data to avoid multicollinearity
                
         # We'll first find the ideal number of components to consider in
         # our analysis.
                        
    pc=PCA(n_components=X.shape[1]).fit(scl_pred)
    pc_var=pc.explained_variance_ratio_
    pc_var_cumsum=np.cumsum(np.round(pc_var, decimals=4)*100)
    print('The ideal number of components is', np.unique(pc_var_cumsum).size)
    mpl.plot(pc_var_cumsum)
    mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/idealnumberofcomponents.png")
    mpl.close()
               
            
         # The ideal number of components is the first with the highest variance
         # cumulative sum. The plot above ilustrates the choice of the components
         # executed on the next line of code
            
    new_pc=PCA(n_components=np.unique(pc_var_cumsum).size).fit_transform(scl_pred)
   
    
    return pd.DataFrame(new_pc)


# Using the function to create the new predictors dataframe

x=Normalize_Scale_and_Reduce(X)



# Handling outliers

mpl.boxplot(x)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/boxplot_predictors.png")
mpl.close()

mpl.boxplot(y)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/boxplot_target.png")
mpl.close()

    """
    Based on the plots shown above, next we have the function to handle all the
    outliers usint the imputation technique
    """

def imputation (data):
        
    n_outliers=[]
    q1=np.quantile(data, 0.25)
    q3=np.quantile(data, 0.75)
    iqr=q3-q1       
    
    for k in data:
        
        # As there's only values beyond the upper whisker. The found
        # outliers will be determined just for this area.
        
        if k>(q3+1.5*iqr):
            n_outliers.append(k)
                 
     # For imputation, it'll be used the median value, since the mean tends
     # to be highly influenced by outliers.
     
    data_imp = np.where(data>=min(n_outliers), data.median(), data)          
    
    return pd.DataFrame(data_imp)
        
        """
        Replacing outliers with the median value.
        """
x[0]=imputation(x[0])
x[2]=imputation(x[2])
y['Y house price of unit area']=imputation(y['Y house price of unit area'])


        """
        Displaying the boxplots after the outliers treatment.
        """

mpl.boxplot(x)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/boxplot_PredictorsImputation.png")
mpl.close()

mpl.boxplot(y)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/RealEstateValuation/boxplot_TargetImputation.png")
mpl.close()


""" BUILDING THE PREDICTION MODEL """


# Splitting the data into train and test sets

x_train, x_test, y_train, y_test = tts(x,y, test_size=0.2, random_state=97)


# A quick look at the shapes of train and test sets

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Training the model

LINREG=lreg()
LINREG.fit(x_train, y_train)
y_pred=LINREG.predict(x_test)


# Presenting the metrics to analyse the performance of the model
 
print('Test set (RMSE):', mean_squared_error(y_test, y_pred, squared=False))

print('NRMSE:', (mean_squared_error(y_test, y_pred, squared=False)/(y.max()-y.min())))

print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))


# Analysing the statistical significance

X_train_=x_train
y_train_=y_train
X_train_=sm.add_constant(X_train_)
Reg=sm.OLS(y_train_, X_train_).fit()
Reg.summary()

        """
        Based on the above summary we notice that the last predictor have not 
        statistical significance (p-value>0.05). Hence, with the conditions
        of our model, it does not affect the target.
        """

# Retraining the model


x1=pd.concat([x[0], x[1], x[2]], axis=1)


# Splitting the data into train and test sets

x1_train, x1_test, y1_train, y1_test = tts(x1,y, test_size=0.2, random_state=97)


# Shapes of the new train and test sets

print(x1_train.shape)
print(y1_train.shape)
print(x1_test.shape)
print(y1_test.shape)


# Reraining the model

LINREG1=lreg()
LINREG1.fit(x1_train, y1_train)
y1_pred=LINREG1.predict(x1_test)


# Presenting the metrics to reanalyse the performance of the model
 
print('Test set (RMSE):', mean_squared_error(y1_test, y1_pred, squared=False))

print('NRMSE:', (mean_squared_error(y1_test, y1_pred, squared=False)/(y.max()-y.min())))

print('Coefficient of determination (R^2):', r2_score(y1_test, y1_pred))
        
    """
    As we can notice, removing the last reduced predictor, the performance of 
    the model is practically the same (from 0.629 to 0.623).
    Hence, we can infer that about 62.3% of the target variance can be explained
    by the predictors variance.
    """

# Reanalysing the statistical significance

X_train_1=x1_train
y_train_1=y1_train
X_train_1=sm.add_constant(X_train_1)
Reg1=sm.OLS(y_train_1, X_train_1).fit()
Reg1.summary()

______________________________________________end___________________________________________________