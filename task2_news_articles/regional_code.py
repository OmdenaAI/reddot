import glob
import pandas as pd
from googletrans import Translator
import numpy as np
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
translator = Translator()
news = pd.read_csv('NONen_interim_01_05_ALL_IN.csv', index_col=False)
def get_full_text(url):
    try:
        req = Request(url,headers=header)
        news_article = BeautifulSoup(urlopen(req), features="html.parser", from_encoding="iso-8859-1").get_text()
    except:
        news_article = 'N/A'
    return news_article
news["text"] = news["SOURCEURL"].apply(get_full_text)
news_df = news.copy()
print(news_df)
news_df['text'] = news_df['text'].apply(translator.translate,dest='en').apply(getattr, args=('text'))
news_df.to_csv("NONen_interim_01_05_DV_IN_with_full_texts.csv", encoding='utf-8')
news_df[news_df['text'] == 'N/A'].count()
