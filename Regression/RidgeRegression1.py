##importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##### GETTING HIGHER OREDER POLYNOMIAL FEATURES ##

def polynomial_dataframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_dframe = pd.DataFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_dframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_sframe[name] to be feature^power
            poly_dframe[name]= feature**power
    return poly_dframe

##LOADING DATA

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])

## OBSERVING HIGH COMPLEXITY MODEL WITH DEGREE 15 ##

poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
l2_small_penalty = 1e-5

from sklearn import linear_model

model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
ridgeModel=model.fit(poly15_data, sales['price'])

ridgeModel.coef_  ### coefficients increase drastically as complexity increases ###

### checking ridge for subset of Data ##

# dtype_dict same as above
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

## SETTING 15DEGREE POLYNOMIAL FOR EACH SET ##

poly15_set1 = polynomial_dataframe(set_1['sqft_living'], 15)
poly15_set2 = polynomial_dataframe(set_2['sqft_living'], 15)
poly15_set3 = polynomial_dataframe(set_3['sqft_living'], 15)
poly15_set4 = polynomial_dataframe(set_4['sqft_living'], 15)

## CREATING MODELS FOR EACH SET with small penalty##
l2_small_penalty=1e-9

set1_model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
ridgeSet1=set1_model.fit(poly15_set1, set_1['price'])

set2_model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
ridgeSet2=set2_model.fit(poly15_set2, set_2['price'])

set3_model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
ridgeSet3=set3_model.fit(poly15_set3, set_3['price'])

set4_model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
ridgeSet4=set4_model.fit(poly15_set4, set_4['price'])

##GETTING COEFFICIENTS FOR EACH MODEL

ridgeSet1.coef_
ridgeSet2.coef_
ridgeSet3.coef_
ridgeSet4.coef_

## plotting coefficient for Model1

plt.plot(poly15_set1['power_15'],set_1['price'],'.',
        poly15_set1['power_15'], ridgeSet1.predict(poly15_set1),'-')

plt.show()

## plotting coefficient for  model2

plt.plot(poly15_set2['power_15'],set_2['price'],'.',
        poly15_set2['power_15'], ridgeSet2.predict(poly15_set2),'-')

plt.show()

## plotting coefficient for model3

plt.plot(poly15_set3['power_15'],set_3['price'],'.',
        poly15_set3['power_15'], ridgeSet3.predict(poly15_set3),'-')

plt.show()

## plotting coefficient for each model4

plt.plot(poly15_set4['power_15'],set_4['price'],'.',
        poly15_set4['power_15'], ridgeSet4.predict(poly15_set4),'-')

plt.show()


## CREATING MODELS FOR EACH SET with large penalty##
l2_large_penalty=1.23e2

set1_model = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
ridgeSet1=set1_model.fit(poly15_set1, set_1['price'])

set2_model = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
ridgeSet2=set2_model.fit(poly15_set2, set_2['price'])

set3_model = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
ridgeSet3=set3_model.fit(poly15_set3, set_3['price'])

set4_model = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)
ridgeSet4=set4_model.fit(poly15_set4, set_4['price'])

## plotting coefficient for each model1##

plt.plot(poly15_set1['power_15'],set_1['price'],'.',
        poly15_set1['power_15'], ridgeSet1.predict(poly15_set1),'-')

plt.show()

## plotting coefficient for each model2##

plt.plot(poly15_set2['power_15'],set_2['price'],'.',
        poly15_set2['power_15'], ridgeSet2.predict(poly15_set2),'-')

plt.show()

## plotting coefficient for each model3##

plt.plot(poly15_set3['power_15'],set_3['price'],'.',
        poly15_set3['power_15'], ridgeSet3.predict(poly15_set3),'-')

plt.show()

## plotting coefficient for each model4##

plt.plot(poly15_set4['power_15'],set_4['price'],'.',
        poly15_set4['power_15'], ridgeSet4.predict(poly15_set4),'-')

plt.show()

## SELECTING PENALTY BY CROSS VALIDATION

train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('E:\ML-data\Regression\wk3_kc_house_test_data.csv\wk3_kc_house_test_data.csv', dtype=dtype_dict)

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    
    residuals = output -(intercept + (slope*input_feature))

    residuals_squared=residuals*residuals
    RSS =residuals_squared.sum()

    return(RSS)

def k_fold_cross_validation(k, l2_penalty, data, output):
    error=0
    for i in xrange(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation_set = data[start:end+1]
        training_set=data[0:start].append(data[end+1:n])
        model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
        model.fit(training_set,training_set[output])
        error += get_residual_sum_of_squares(poly15set,validation_set[output],model.intercept_,model.slope_)
    average_validation_error=error/k
    return [average_validation_error]
    
## looping on various values of penalty

poly15set = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)

for l2_penalty in range(np.logspace(1, 7, num=13)):
      error = k_fold_cross_validation(10, l2_penalty, poly15set,poly15set['price'])  
      print l2_penalty + " : " +  error
