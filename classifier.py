
import pandas as pd

import nltk
# Import NLTK to use its functionalities on texts
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# We will visualize the messages with a word cloud
from wordcloud import WordCloud

# Import Conter Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

# Import Tf-idf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the Label Encoder
from sklearn.preprocessing import LabelEncoder

# Import the train test split
from sklearn.model_selection import train_test_split

# To evaluate our model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

# I will keep the resulting plots
# %matplotlib inline

# Enable Jupyter Notebook's intellisense
# %config IPCompleter.greedy=True

data = pd.read_csv('data3_revised.csv')

#Display firt five rows
print(data.head())

# Display the summary statistics
print(data.describe())

pd.set_option('display.max_rows', None)

dataframe = pd.DataFrame(data)
#print(dataframe['category', 'label'])
print(dataframe.columns.tolist())

# Print the info
print(data.info())

#print(data['category'].value_counts())

#print(data['category'].value_counts())

#print()

# Print the proportions of each category
print(data['category'].value_counts(normalize=True))

# Make the letters lower case and tokenize the words
tokenized_messages = data['abstract'].str.lower().apply(word_tokenize)

# Define a function to returns only alphanumeric tokens
def alpha(tokens):
# This function removes all non-alphanumeric characters
    alpha = []
    for token in tokens:
        if str.isalpha(token) or token in ['n\'t','won\'t']:
            if token=='n\'t':
                alpha.append('not')
                continue
            elif token == 'won\'t':
                alpha.append('wont')
                continue
            alpha.append(token)
    return alpha

# Apply our function to tokens
tokenized_messages = tokenized_messages.apply(alpha)

def remove_stop_words(tokens):
#This function removes all stop words in terms of nltk stopwords
    no_stop = []
    for token in tokens:
        if token not in stopwords.words('english'):
            no_stop.append(token)
    return no_stop

# Apply our function to tokens
tokenized_abstract = tokenized_messages.apply(remove_stop_words)

def lemmatize(tokens):
    ## This function lemmatize the messages
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # Create the lemmatized list
    lemmatized = []
    for token in tokens:
            # Lemmatize and append
            lemmatized.append(lemmatizer.lemmatize(token))
    return " ".join(lemmatized)

# Apply our function to tokens
tokenized_messages = tokenized_abstract.apply(lemmatize)

# Replace the columns with tokenized messages
data['abstract'] = tokenized_messages

# Select the features and the target
# X = data['abstract']
# y = data['category']
X_train = data['abstract']
y_train = data['category']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)

X_train.head()

count_vect = CountVectorizer()
count_vect.fit(X_train)
count_vect_transform = count_vect.transform(X_train)

count_vect_transform

count_vect_transform.toarray()

pd.DataFrame(count_vect_transform.toarray(), columns = count_vect.get_feature_names()).head()

tfidf_vect = TfidfTransformer()
X_train_tfidf = tfidf_vect.fit(count_vect_transform)
X_train_tfidf_transform = X_train_tfidf.transform(count_vect_transform)

pd.DataFrame(X_train_tfidf_transform.toarray(), columns = count_vect.get_feature_names()).head(25)

clsfy = MultinomialNB().fit(X_train_tfidf_transform, y_train)

print(clsfy.predict(count_vect.transform(["History of Present IllnessThe patient presents with a new problem. Patient is a 63 year old with a past medical history of stroke with residual right weakness  depression  htnwho presents with dizziness  right face numb. History is per patient  chart. Notes from other providers since admission reviewed  labs reviewed imaging results reviewed  medication list reviewed. Patient has residual right face droop from old stroke. Yesterday he felt dizzy and numb on right face.This is better now. No inciting  alleviating  exacerbating factors. Patient denies smoking  drinking alcohol  or using illicit drugs. Family history reviewedand noncontributory to this presentation..Cardiovascular: No chest pain.Gastrointestinal: No nausea  No vomiting  No diarrhea.Genitourinary: No dysuria.Integumentary: No rash.Musculoskeletal: No muscle pain.Neurologic: Numbness  No abnormal balance  No confusion  No tingling  No headache.Psychiatric: no mood change  No hallucinations."])))

print(clsfy.predict(count_vect.transform([".Physician Progress Notes * Final Report * Result type: Date/Time of Service: Result status: Result title: Contributed By: Verified by:Encounter info: .Physician Progress Notes June 13, 2021 11:48 PDT Auth (Verified)all results reviewed and discussed with ptinternal medicine progress note DABAS MD, RIDHIMA on June 13, 2021 11:51 MST DABAS MD, RIDHIMA on June 13, 2021 11:51 MST84516491, BTMC, Observation, 06/12/2021 - 06/14/2021 Plan assessment and plan- 1. Chest pain,-troponin negative EKG negative appears typical as patient feels the pain is similar to the last episode when he had stents placed 2 years ago. Patient is continuing to be symptomatic trial of oral nitroglycerin. We will get a stat EKG cardiology consulted. Continue medical management. We will follow 2. CAD - Coronary artery disease -continue medical management already on aspirin statin as needed nitroglycerin andmorphine 3. Chronic back pain -Resume home narcotic medications, patient on scheduled oxycodone at home which is continuedat home dose avoid more narcotics if not needed diet-cardio activity-as tolerated code status-full PT/CM-no needs TUTRONE, HERBERT NEIL - 197948 4. Hypertension -blood pressure slightly elevated could be secondary to pain continue to monitor add medications astolerated5. COPD - Chronic obstructive pulmonary disease Inhalers as needed does not appear to be exacerbateddvt/gi prophyalxis Lovenox tolerating oralall questions answered and d.w pt/RNDISPO - dependig on clinical course awaiting cardiologyPrinted by:Printed on:Completed Action List:* Perform by DABAS MD, RIDHIMA on June 13, 2021 11:51 MST*Sign by DABAS MD, RIDHIMA on June 13, 2021 11:51 MST* VERIFY by DABAS MD, RIDHIMA on June 13, 2021 11:51 MSTValladares, Luz09/07/2021 11:36 PDTPage 5 of 6.Physician Progress Notes* Final Report *Result type:Date/Time of Service:Result status:Result title:Contributed By:Verified by:Encounter info:Printed by:Printed on:.Physician Progress NotesJune 13, 2021 11:48 PDTAuth (Verified)TUTRONE, HERBERT NEIL - 197948internal medicine progress noteDABAS MD, RIDHIMA on June 13, 2021 11:51 MSTDABAS MD, RIDHIMA on June 13, 2021 11:51 MST84516491, BTMC, Observation, 06/12/2021 - 06/14/2021Valladares, Luz09/07/2021 11:36 PDTPage 6 of 6"])))

print(clsfy.predict(count_vect.transform(["Pyhysician Progress Notes Result type: .Physician Progress Notes June 13 This is an abnormal routine- awake and drowsy EEG due to diffuse mildbackground slowing consistent with encephalopathy, possible "])))

import pickle

pickle.dump(count_vect,open('count_vect1.pkl', 'wb'))
pickle.dump(clsfy, open('lf1.pkl', 'wb'))

# pip install -U scikit-learn

# pip install scikit-learn==0.22.2.post1