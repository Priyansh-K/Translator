import re
import heapq
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
from typing import List

def summarize_and_translate(text: str, target_lang: str = "ne") -> List[str]:

    article_text = text.replace("\n", " ")
    article_text = re.sub(r"\[[0-9]*\]", " ", article_text)
    article_text = re.sub(r"\s+", " ", article_text)

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

    summary_sentences_1 = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary_1 = " ".join(summary_sentences_1)
    
    from nltk.corpus import stopwords
    def lsaSummarize(text):

        sentences = sent_tokenize(text)

        stop_words = stopwords.words('english')

        clean_sentences = [sent.lower() for sent in sentences]

        # Convert the cleaned sentences to a TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)

        # Use TruncatedSVD to reduce the dimensionality of the TF-IDF matrix
        svd = TruncatedSVD(n_components=5, random_state=0)
        svd_matrix = svd.fit_transform(tfidf_matrix)

        scores = np.sum(svd_matrix, axis=1)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        summary = ' '.join([ranked_sentences[i][1] for i in range(3)])

        return summary

    summary_2 = lsaSummarize(text)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    input_ids = tokenizer.encode(text, return_tensors='pt')

    summary_ids = model.generate(input_ids, max_length=50, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary_3 = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    summaries = [summary_1, summary_2, summary_3]
    scores = {}
    for i in range(3):
        for j in range(i+1,3):
            summary1_tokens = word_tokenize(summaries[i])
            summary2_tokens = word_tokenize(summaries[j])

            # Calculate the Jaccard similarity score
            intersection = set(summary1_tokens).intersection(summary2_tokens)
            union = set(summary1_tokens).union(summary2_tokens)
            jaccard_score = len(intersection) / len(union)

            scores[(i,j)] = jaccard_score

    max_score = max(scores.values())

    best_summaries = []
    for key, value in scores.items():
        if value == max_score:
            best_summaries.append(key[0])
            best_summaries.append(key[1])

    summary = summaries[best_summaries[0]]

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

