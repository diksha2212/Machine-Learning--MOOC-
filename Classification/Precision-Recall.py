## IMPORT LIBRARIES
import graphlab
from __future__ import division
import numpy as np
graphlab.canvas.set_target('ipynb')

##LOADING DATA

products = graphlab.SFrame('amazon_baby.gl/')

##EXTRACT WORD COUNT AND SENTIMENTS

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

# Remove punctuation.
review_clean = products['review'].apply(remove_punctuation)

# Count words
products['word_count'] = graphlab.text_analytics.count_words(review_clean)

# Drop neutral sentiment reviews.
products = products[products['rating'] != 3]

# Positive sentiment to +1 and negative sentiment to -1
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

##TRAIN-TEST SPLIT

train_data, test_data = products.random_split(.8, seed=1)

##TRAINING CLASSIFIER

model = graphlab.logistic_classifier.create(train_data, target='sentiment',
                                            features=['word_count'],
                                            validation_set=None)
                                            
##ACCURACY

accuracy= model.evaluate(test_data, metric='accuracy')['accuracy']
print "Test Accuracy: %s" % accuracy

##CONFUSION MATRIX

confusion_matrix = model.evaluate(test_data, metric='confusion_matrix')['confusion_matrix']

##PRECISION
precision = model.evaluate(test_data, metric='precision')['precision']

##RECALL
recall = model.evaluate(test_data, metric='recall')['recall']

##PRECISION-RECALL TRADEOFF

from graphlab import SArray
##APPLYING THRESHOLD

def apply_threshold(probabilities, threshold):
    ### YOUR CODE GOES HERE
    # +1 if >= threshold and -1 otherwise.
    output=[]
    for i in range(len(probabilities))  :
        if(probabilities[i]>= threshold):
            output.append(+1)
        else:
            output.append(-1)       
    output=SArray(data=output, dtype=int)
    return output
    
probabilities = model.predict(test_data, output_type='probability')
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)


from graphlab import SArray
# Threshold = 0.5
predictions_with_default_threshold = SArray(data=predictions_with_default_threshold, dtype=int)
precision_with_default_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_default_threshold)

                                                  
recall_with_default_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_default_threshold)

# Threshold = 0.9
predictions_with_high_threshold=SArray(data=predictions_with_high_threshold, dtype=int)
precision_with_high_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_high_threshold)
recall_with_high_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_high_threshold)
                                        
  ##PLOTTING PRECISION RECALL CURVE
  
  import matplotlib.pyplot as plt
%matplotlib inline

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
