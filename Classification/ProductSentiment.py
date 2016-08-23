##importing libraries
import graphlab

##LOADING PRODUCT DATA

products = graphlab.SFrame('amazon_baby.gl/')

##BUILD WORD COUNT VECTOR FOR EACH REVIEW
products['word_count'] = graphlab.text_analytics.count_words(products['review'])

##DEFINING POSITIVE AND NEGATIVE SENTIMENT

#ignore all 3* reviews
products = products[products['rating'] != 3]

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4

##TRAIN SENTIMENT CLASSIFIER

train_data,test_data = products.random_split(.8, seed=0)
sentiment_model = graphlab.logistic_classifier.create(train_data,target='sentiment',features=['word_count'], validation_set=test_data)

##EVALUATE SENTIMENT MODEL
sentiment_model.evaluate(test_data, metric='roc_curve')
sentiment_model.show(view = 'Evaluation')

#3 PREDICT SENTIMENT FOR GIRAFFE PRODUCT
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews,
                                                                 output_type = 'probability')
                                                                 
##TRAIN USING FEW WORDS FROM REVIEWS

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

def awesome_count(dict):
    if 'awesome' in dict:
        return dict['awesome']
    else:
        return 0
    
def great_count(dict):
    if 'great' in dict:
        return dict['great']
    else:
        return 0
    
def fantastic_count(dict):
    if 'fantastic' in dict:
        return dict['fantastic']
    else:
        return 0
    
def amazing_count(dict):
    if 'amazing' in dict:
        return dict['amazing']
    else:
        return 0    
    
def love_count(dict):
    if 'love' in dict:
        return dict['love']
    else:
        return 0   
    
def horrible_count(dict):
    if 'horrible' in dict:
        return dict['horrible']
    else:
        return 0    
    
def bad_count(dict):
    if 'bad' in dict:
        return dict['bad']
    else:
        return 0    
    
def terrible_count(dict):
    if 'terrible' in dict:
        return dict['terrible']
    else:
        return 0    
    
def awful_count(dict):
    if 'awful' in dict:
        return dict['awful']
    else:
        return 0    
    
def wow_count(dict):
    if 'wow' in dict:
        return dict['wow']
    else:
        return 0 
    
def hate_count(dict):
    if 'hate' in dict:
        return dict['hate']
    else:
        return 0  
        
products['awesome'] = products['word_count'].apply(awesome_count)
products['great'] = products['word_count'].apply(great_count)
products['fantastic'] = products['word_count'].apply(fantastic_count)
products['amazing'] = products['word_count'].apply(amazing_count)
products['love'] = products['word_count'].apply(love_count)
products['horrible'] = products['word_count'].apply(horrible_count)
products['bad'] = products['word_count'].apply(bad_count)
products['terrible'] = products['word_count'].apply(terrible_count)
products['awful'] = products['word_count'].apply(awful_count)
products['wow'] = products['word_count'].apply(wow_count)
products['hate'] = products['word_count'].apply(hate_count)

##TRAINING MODEL USING SELECTED WORDS COUNT

selected_words_model=graphlab.logistic_classifier.create(train_data,
                                                        target='sentiment', features = selected_words,validation_set = test_data)
 
 ##ANALYSING COEFFICIENTS                                                       
selected_words_model['coefficients'].sort('value')
                                                     
##EVALUATING
selected_words_model.evaluate(test_data)

##PREDICTING DIAPER REVIEW ACCURACY USING SELECTED WORDS MODEL

selected_words_model.predict(diaper_reviews[0:1], output_type='probability')
