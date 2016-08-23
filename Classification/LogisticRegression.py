## IMPORT GRAPHLAB (USING GRAPHLAB AND SFRAME)
import math
import graphlab

##LOADING DATA
products = graphlab.SFrame('amazon_baby_subset.gl/')

##COUNTING NO.OF POSITIVE AND NEAGATIVE REVIEWS IN THE DATASET
print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

# Reads the list of most frequent words TO ANALYSE SENTIMENT
import json
with open('important_words.json', 'r') as f: 
    important_words = json.load(f)
important_words = [str(s) for s in important_words]

##CLEANING REVIEWS 

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
    
    ##FUNCTION TO CONVERT DATA TO NUMPY FORM
    
    def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)
  
  ##GET FEATURE MATRIX AND LABELS
  feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment') 
  
  
'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    # YOUR CODE HERE
    scores=np.dot(feature_matrix,coefficients)
    predictions = np.zeros(shape = len(scores))
    # Compute P(y_i = +1 | x_i, w) using the link function
    # YOUR CODE HERE
    for i in range (len(scores)):
     predictions[i] =np.array([1/(1+ np.exp(-scores[i]))])
     
    # return predictions
    return predictions
  
  ##HELPER METHOD
  def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(feature,errors)
    
    # Return the derivative
    return derivative
    
  ##COMPUTING LOG
  
  def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp
    
    
    ##logistic regression implementation
    
    from math import sqrt

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # YOUR CODE HERE
        predictions = predict_probability(feature_matrix,coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = feature_derivative(errors,feature_matrix[:,j])
            
            # add the step size times the derivative to the current coefficient
            ## YOUR CODE HERE
            coefficients[j]+=step_size*derivative
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
    
    ##GETTING COEFFICIENTS
    coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),step_size=1e-7, max_iter=301)
    scores = np.dot(feature_matrix, coefficients)
    
    
    ##FUNCTION TO GET PREDICTIONS
    def class_predictions(feature_matrix, coefficients):
 scores = np.dot(feature_matrix, coefficients)
 sentiments = np.zeros(shape=len(scores))
 for i in range(len(scores)):
  if(scores[i]>0):
    sentiments[i]=1
  else:
   sentiments[i]=-1
 return sentiments

sentiments= class_predictions(feature_matrix,coefficients)
positive_count=0
negative_count=0
for i in range(len(sentiments)):
    if(sentiments[i]==1):
      positive_count+=1
    else:
      negative_count+=1  
print '# of positive reviews =', positive_count
print '# of negative reviews =', negative_count
                                   step_size=1e-7, max_iter=301)
