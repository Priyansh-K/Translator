from typing import List
import bs4 as bs
import nltk
import heapq
import requests
import re

def summarize_and_translate(text: str, target_lang: str = "ne") -> List[str]:
    # Remove Square Brackets and Extra Spaces
    article_text = text.replace("\n", " ")
    article_text = re.sub(r"\[[0-9]*\]", " ", article_text)
    article_text = re.sub(r"\s+", " ", article_text)

    # Remove special characters and digits
    formatted_article_text = re.sub("[^a-zA-Z]", " ", article_text)
    formatted_article_text = re.sub(r"\s+", " ", formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words("english")
    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequency

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(" ")) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = " ".join(summary_sentences)

    # Translate summary to Nepali
    translated_text = ""
    length = len(summary)
    parts = [summary[i:i+500] for i in range(0, length, 500)]
    for part in parts:
        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": part,
            "langpair": "en|ne",
        }
        response = requests.get(url, params=params)
        data = response.json()
        translated_text += data["responseData"]["translatedText"]

    return [translated_text]


    #translated_sentences = []
    #for sentence in summary_sentences:
        #translator = Translator(to_lang=target_lang)
        #words = sentence.split(" ")
        #translated_words = []
        #for word in words:
        #    translation = translator.translate(word)
        #    translated_words.append(translation)
        #translated_sentence = " ".join(translated_words)
        #translated_sentences.append(translated_sentence)

    #return translated_sentences

