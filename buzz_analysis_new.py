import pandas as pd
pd.options.mode.chained_assignment = None

# Load the df
df_main = pd.read_csv('lingbuzz_2_7448.csv')

# Drop nan entries
df_main.dropna(subset=['Title'], inplace=True)


#%%
# These are the labels to use
disciplines = ['phonology', 'morphology', 'syntax', 'semantics']

# Creating a new df with the relevant information
to_train = df_main[['Title', 'Abstract', 'Keywords']]

# Filling the empty abstracts
to_train['Abstract'] = to_train['Abstract'].fillna('')

# Creating columns for the labels and giving them values 0 and 1
for word in disciplines:
    to_train[word] = to_train['Keywords'].apply(lambda x: 1 if word in x else 0)

# Combining the text in titles and abstracts
to_train['context'] = to_train['Title'] + '. ' + to_train['Abstract']

# Reordering the columns and droping the unnecesary ones
to_train = to_train[['context', 'phonology', 'morphology', 'syntax',
       'semantics']]

# Dropping manuscripts that did not use one of the relevant keywords
to_train = to_train.drop(to_train[(to_train[disciplines] == 0).all(axis=1)].index)

#%%

#to_train[['phonology', 'morphology', 'syntax', 'semantics']].sum().plot.bar()

#%%
rows_to_delete = to_train[to_train['syntax'] == 1].head(3500).index

to_train.drop(rows_to_delete, inplace=True)

#%%

to_train[['phonology', 'morphology', 'syntax', 'semantics']].sum().plot.bar()

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=None,  # Set the maximum number of features (words) to consider. Use None to consider all words.
    stop_words='english',  # Remove English stopwords during vectorization.
    lowercase=True,  # Convert all words to lowercase during vectorization.
)

X_features = vectorizer.fit_transform(to_train['context'])#.toarray()


#%%
from sklearn.model_selection import train_test_split

y = to_train[['phonology', 'morphology', 'syntax', 'semantics']]

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size = 0.2, random_state=42)

#%%
#import skmultilearn
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance

classifier_1 = BinaryRelevance(MultinomialNB())

classifier_1.fit(X_train, y_train)

y_pred = classifier_1.predict(X_test)#.toarray()

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



accuracy_1 = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')  # For multi-label classification
recall = recall_score(y_test, y_pred, average='micro')        # For multi-label classification
f1 = f1_score(y_test, y_pred, average='micro')                # For multi-label classification


#%%
from sklearn.metrics import hamming_loss
hamming = hamming_loss(y_test, y_pred)

#%%

vectorizer.get_feature_names_out()

#%%
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words='english', lowercase=True)

experiment_vec = vec.fit_transform(to_train['context'])

#%%

enter_abstract = input('Copy your abstract here: ')

experiment = vec.transform([enter_abstract])

exp_pred = classifier_1.predict(experiment).toarray()

cats = ['phonology', 'morphology', 'syntax', 'semantics']
output = []


for num in range(4):
    if exp_pred[0, num] == 1:
        output.append(cats[num])

print('This is an abstract on ' + ', '.join(output))
        



