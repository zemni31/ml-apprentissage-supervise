import pandas as pd
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Chargement des données
DATA_DIR = r"C:\Users\MSI\Downloads\Atelier 3\Atelier 3\DB\DB"
chemin = os.path.join(DATA_DIR, "spam.csv")
df = pd.read_csv(chemin, encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Stopwords et stemmer chargés une seule fois
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
STEMMER = Stemmer()

# 2. Nettoyage et normalisation d'un message
def process(text):
    text = text.lower()
    text = ''.join([t for t in text if t not in string.punctuation])
    text = [t for t in text.split() if t not in STOP_WORDS]
    return [STEMMER.stem(t) for t in text]

# --- Démonstration TF-IDF (facultative)
# print(process('It\'s holiday and we are playing cricket. Jeff is playing very well!!!'))
# print(df['message'][:20].apply(process))
# tfidfv = TfidfVectorizer(analyzer=process)
# data = tfidfv.fit_transform(df['message'])
# mess = df.iloc[2]['message']
# print(mess)
# print(tfidfv.transform([mess]))
# j = tfidfv.transform([mess]).toarray()[0]
# print('index\tidf\ttfidf\tterm')
# for i in range(len(j)):
#     if j[i] != 0:
#         print(i, format(tfidfv.idf_[i], '.4f'), format(j[i], '.4f'),
#               tfidfv.get_feature_names_out()[i], sep='\t')

# 3. Pipeline : TF-IDF puis Naïve Bayes
spam_filter = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)),
    ('classifier', MultinomialNB())
])

# 4. Découpage 80 % apprentissage / 20 % test
x_train, x_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.20, random_state=21
)

# 5. Entraînement
spam_filter.fit(x_train, y_train)

# 6. Prédictions sur le jeu de test
predictions = spam_filter.predict(x_test)

# 7. Question 2a : erreurs
count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1
        print(f"Message : {x_test.iloc[i]}")
        print(f"  Vrai label : {y_test.iloc[i]} | Prédit : {predictions[i]}")
print('Total number of test cases', len(y_test))
print('Number of wrong predictions', count)

# 8. Question 2b : vérification d'un message
message_test = "Your cash-balance is currently 500 pounds - to maximize your cash-in now, send COLLECT to 83600."
print("Prédiction :", spam_filter.predict([message_test])[0])
probas = spam_filter.predict_proba([message_test])[0]
print("Probabilités :", {str(c): round(float(p), 4) for c, p in zip(spam_filter.classes_, probas)})

# 9. Question 2d : précision, rappel, F1-score
print(classification_report(y_test, predictions, digits=4))