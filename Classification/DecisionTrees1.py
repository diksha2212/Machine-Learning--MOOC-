##import libraries

import graphlab
graphlab.canvas.set_target('ipynb')

##Loading data

loans = graphlab.SFrame('lending-club-data.gl/')

##DEFINING SAFE AND RISKY LOANS
# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

##DEFINE FEATURES AND TARGET
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

##ANALYSING DATA FOR SAFE AND RISKY

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)

##TRAIN-VALIDATION SPLIT

train_data, validation_data = loans_data.random_split(.8, seed=1)

##TRAINING DECISION TREE
decision_tree_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                target = target, features = features)
                                
##SETTING SMALL MODEL, DEPTH=2

small_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 2)
                   
  ##PREDICTING
decision_tree_model.predict(sample_validation_data)

##PREDICTING PROBABILITY
decision_tree_model.predict(sample_validation_data,output_type ='probability')

small_model.predict(sample_validation_data,output_type ='probability')

##EVALUATING BOTH MODELS

print small_model.evaluate(train_data)['accuracy']
print decision_tree_model.evaluate(train_data)['accuracy']

##evaluation on validation data
print decision_tree_model.evaluate(validation_data)['accuracy']

##TRAINING AND EVALUATING A BIG MODEL

big_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 10)
                   
print big_model.evaluate(train_data)['accuracy']
print big_model.evaluate(validation_data)['accuracy']
