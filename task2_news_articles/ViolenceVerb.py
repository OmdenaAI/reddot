import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

from afinn import Afinn
afinn = Afinn()

df = pd.read_csv('english_5k_label_final.csv')
df = df[df['lab_final']!= 'Irrelevant']
df['summary'] = df['summary'].apply(lambda x: x.lower())
df['doc'] = df['summary'].apply(nlp)


def get_verb(doc):
    verbs = []
    for token in doc:
        if token.pos_ == 'VERB':
            verbs.append(token.lemma_)          
    return verbs

def negative_verbs(verb):
    if afinn.score(verb) <0:
        return verb

df['verbs'] = df['doc'].apply(lambda x: get_verb(x))
df['negative_verbs'] = df['verbs'].apply(lambda x: [i for i in x if afinn.score(i)<-1])


