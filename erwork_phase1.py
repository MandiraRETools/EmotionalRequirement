

!pip install requests beautifulsoup4 textblob
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

!pip install google-play-scraper

import google_play_scraper

app_id1 = 'com.pazugames.avatarworld'
app_id2= 'com.jetstartgames.chess'
app_id3='com.PepiPlay.KingsCastle'
app_id4='easy.sudoku.puzzle.solver.free'
app_id5='games.vaveda.militaryoverturn'
app_id17= 'org.coursera.android'
app_id7= 'com.udemy.android'
app_id8= 'com.linkedin.android.learning'
app_id9='com.phonegap.rxpal'
app_id10='com.apollo.patientapp'
app_id11= 'com.remente.app'
app_id12= 'com.elevatelabs.geonosis'
app_id13= 'com.smilingmind.app'
app_id15= 'com.samsung.android.oneconnect'
app_id14='com.google.android.apps.chromecast.app'
app_id16='io.homeassistant.companion.android'
app_id='com.Nanali.ForestIsland'

from google_play_scraper import Sort
from google_play_scraper.constants.element import ElementSpecs
from google_play_scraper.constants.regex import Regex
from google_play_scraper.constants.request import Formats
from google_play_scraper.utils.request import post

import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import json
from time import sleep
from typing import List, Optional, Tuple

MAX_COUNT_EACH_FETCH = 199


class _ContinuationToken:
    __slots__ = (
        "token",
        "lang",
        "country",
        "sort",
        "count",
        "filter_score_with",
        "filter_device_with",
    )

    def __init__(
        self, token, lang, country, sort, count, filter_score_with, filter_device_with
    ):
        self.token = token
        self.lang = lang
        self.country = country
        self.sort = sort
        self.count = count
        self.filter_score_with = filter_score_with
        self.filter_device_with = filter_device_with


def _fetch_review_items(
    url: str,
    app_id: str,
    sort: int,
    count: int,
    filter_score_with: Optional[int],
    filter_device_with: Optional[int],
    pagination_token: Optional[str],
):
    dom = post(
        url,
        Formats.Reviews.build_body(
            app_id,
            sort,
            count,
            "null" if filter_score_with is None else filter_score_with,
            "null" if filter_device_with is None else filter_device_with,
            pagination_token,
        ),
        {"content-type": "application/x-www-form-urlencoded"},
    )
    match = json.loads(Regex.REVIEWS.findall(dom)[0])

    return json.loads(match[0][2])[0], json.loads(match[0][2])[-2][-1]


def reviews(
    app_id: str,
    lang: str = "en",
    country: str = "us",
    sort: Sort = Sort.MOST_RELEVANT,
    count: int = 100,
    filter_score_with: int = None,
    filter_device_with: int = None,
    continuation_token: _ContinuationToken = None,
) -> Tuple[List[dict], _ContinuationToken]:
    sort = sort.value

    if continuation_token is not None:
        token = continuation_token.token

        if token is None:
            return (
                [],
                continuation_token,
            )

        lang = continuation_token.lang
        country = continuation_token.country
        sort = continuation_token.sort
        count = continuation_token.count
        filter_score_with = continuation_token.filter_score_with
        filter_device_with = continuation_token.filter_device_with
    else:
        token = None

    url = Formats.Reviews.build(lang=lang, country=country)

    _fetch_count = count

    result = []

    while True:
        if _fetch_count == 0:
            break

        if _fetch_count > MAX_COUNT_EACH_FETCH:
            _fetch_count = MAX_COUNT_EACH_FETCH

        try:
            review_items, token = _fetch_review_items(
                url,
                app_id,
                sort,
                _fetch_count,
                filter_score_with,
                filter_device_with,
                token,
            )
        except (TypeError, IndexError):
            #funnan MOD start
            token = continuation_token.token
            continue
            #MOD end

        for review in review_items:
            result.append(
                {
                    k: spec.extract_content(review)
                    for k, spec in ElementSpecs.Review.items()
                }
            )

        _fetch_count = count - len(result)

        if isinstance(token, list):
            token = None
            break

    return (
        result,
        _ContinuationToken(
            token, lang, country, sort, count, filter_score_with, filter_device_with
        ),
    )


def reviews_all(app_id: str, sleep_milliseconds: int = 0, **kwargs) -> list:
    kwargs.pop("count", None)
    kwargs.pop("continuation_token", None)

    continuation_token = None

    result = []

    while True:
        _result, continuation_token = reviews(
            app_id,
            count=MAX_COUNT_EACH_FETCH,
            continuation_token=continuation_token,
            **kwargs
        )

        result += _result

        if continuation_token.token is None:
            break

        if sleep_milliseconds:
            sleep(sleep_milliseconds / 1000)

    return result

reviews_count = 1000

result = []
continuation_token = None


with tqdm(total=reviews_count, position=0, leave=True) as pbar:
    while len(result) < reviews_count:
        new_result, continuation_token = reviews(
            app_id,
            continuation_token=continuation_token,
            lang='en', #The language of review
            sort=Sort.MOST_RELEVANT,
            filter_score_with=None,
            count=199 #No need to change this
        )
        if not new_result:
            break
        result.extend(new_result)
        pbar.update(len(new_result))
df = pd.DataFrame(result)

df.head()
df.to_csv("google_play_reviews.csv", index=False)

"""Now code to mapping reviews to features using LLM."""

!pip install -q -U langchain langchain_core together
!pip install langchain-together
from google.colab import userdata
!pip install -q -U langchain langchain_core
from langchain_together.llms import Together

from langchain_core.prompts import ChatPromptTemplate

rag_template = """\

    You are an AI assistant for requirements analysis.
    Analyse the review and map it to the corresponding product feature (functional feature) mentioned in the review.

    {review}

    The reviews are of a product named Remente: Well Being and Self Help. Some details of the product are as follows:

With Remente you can improve the quality of your life, rebalance it and increase your well-being. We help you discover what is important to you and make sure you can achieve it, giving you the tools to live your life to the fullest.

We help you focus your life and improve your well-being
We help you define, monitor, achieve goals and build good habits
We help you better manage your life with courses, tools and advice

Your digital life coach at home, at work and at school.
Thanks to recent studies in Psychology, Neuroscience and Mental Training, we have created an app to improve your personal development with dedicated courses and tools to help you improve every day. Do you want to easily achieve your goals, reduce stress, sleep better, be happier or simply get inspiration on how to make things work better? We can help you!

Why should you train your mind?
Just as you train for your physical well-being, training your mind is also important.
To help you, we created Remente, a preventive tool for the well-being and resilience of your mind.

With Remente, you can exercise self-awareness, prevent stress, train your mind, improve sleep, set and achieve goals, find motivation, become more efficient and learn to prioritize things. You will find many training tools: leadership, communication, self-awareness, learning to make the right decisions, time management and much more

The minds behind Remente:
In the past, we supported the Swedish memory team, winning the World Memory Championships, and trained professional athletes such as Annika Sörenstam, Magdalena Forsberg and Nick Faldo. We are supported by a team of world-class experts in the fields of psychology, coaching and mental training who support the research of internet psychiatry.

Some examples of the content offered:
- Stress Management
- Efficiency
- Sleep Cycles
- Memory Training
- Goal Achievement
- Self-Awareness
- Decision Making
- Time Management
- Leadership
- Self-Awareness
- Communication

Output the product features corresponding to each review as:
Review -> Mapped Features
List only the associated features. Do not include any additional information in your response.
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

!pip install together

from together import Together

client = Together(api_key='.........')

import csv
import time
sleep_time = 5

input_csv = 'Remente_selected_reviews.csv'
output_csv = 'Remente_mapping_feature.csv'
with open(input_csv, 'r', encoding='latin-1') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile: # Changed encoding to latin-1 for infile and utf-8 for outfile
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["Review", "Feature"])  # Write header row

        # Skip header row if it exists
        #next(reader, None)

        for row in reader:
            review = row['Reviews']
            #component = row[0]
            #user_query = row[1]  # Assuming query is in the first column
            #combined_content = retrieve_relevant_docs(user_query)
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Based on the given prompt the user query "},
                    {"role": "user", "content": f"Here is the {review + rag_template}"}
                ]
            )

            response = completion.choices[0].message.content
            writer.writerow([review, response])
            time.sleep(sleep_time)
            #print(f"Processed query: {user_query}")
from google.colab import files
#files.download('ForestIsland_mapping_feature.csv')

"""Do first clustering then sort out reviews"""

!pip install pandas nltk scikit-learn sentence-transformers hdbscan

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)

# 1. Load Data and Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'Reviews' not in df.columns:
        raise ValueError("The CSV must have a column named 'Reviews'")
    df['cleaned_review'] = df['Reviews'].apply(clean_text)
    df.drop_duplicates(subset='cleaned_review', inplace=True)
    return df

# 2. Text Embedding
def get_sentence_embeddings(texts):
  model = SentenceTransformer('all-mpnet-base-v2')
  embeddings = model.encode(texts)
  return embeddings

# Alternative if sentence transformer doesn't work
def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()

# 3. Clustering
def cluster_reviews(embeddings, min_cluster_size=50):
    cluster = HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = cluster.fit_predict(embeddings)
    return clusters

# --- Main Clustering Function ---
def main_clustering(file_path):
    # 1. Load and Clean Data
    try:
        df = load_and_clean_data(file_path)
    except FileNotFoundError:
      print("File not found, please verify file path.")
      return
    except ValueError as e:
      print(e)
      return

    # 2. Text Embedding
    embeddings = get_sentence_embeddings(df['cleaned_review'].tolist())
    # If sentence transformer fails try:
    # embeddings = get_tfidf_embeddings(df['cleaned_review'].tolist())

    # 3. Clustering
    clusters = cluster_reviews(embeddings, min_cluster_size=50)

    # Add clusters to DataFrame
    df['cluster'] = clusters

    #Print number of unique clusters and the number of outlier clusters
    print(f"Number of unique clusters found: {len(df['cluster'].unique())}")
    print(f"Number of outlier reviews {len(df[df['cluster'] == -1])}")

    return df

# Example Usage
if __name__ == "__main__":
    file_path = 'AvtarReviews_csv.csv'  # Replace with the actual path to your CSV file
    clustered_df = main_clustering(file_path)

    if clustered_df is not None:
      print(clustered_df.head())  # Print the first few rows of the clustered data
      # You can save the dataframe to CSV file here if needed
      clustered_df.to_csv('clustered_reviews.csv', index = False)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)

# 1. Load Data and Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'Reviews' not in df.columns:
        raise ValueError("The CSV must have a column named 'Reviews'")
    df['cleaned_review'] = df['Reviews'].apply(clean_text)
    df.drop_duplicates(subset='cleaned_review', inplace=True)
    return df

# 2. Text Embedding
def get_sentence_embeddings(texts):
  model = SentenceTransformer('all-mpnet-base-v2')
  embeddings = model.encode(texts)
  return embeddings

# Alternative if sentence transformer doesn't work
def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()

# 3. Clustering
def cluster_reviews(embeddings, min_cluster_size=50):
    cluster = HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = cluster.fit_predict(embeddings)
    return clusters

# --- Main Clustering Function ---
def main_clustering(file_path):
    # 1. Load and Clean Data
    try:
        df = load_and_clean_data(file_path)
    except FileNotFoundError:
      print("File not found, please verify file path.")
      return
    except ValueError as e:
      print(e)
      return

    # 2. Text Embedding
    embeddings = get_sentence_embeddings(df['cleaned_review'].tolist())
    # If sentence transformer fails try:
    # embeddings = get_tfidf_embeddings(df['cleaned_review'].tolist())

    # 3. Clustering
    clusters = cluster_reviews(embeddings, min_cluster_size=50)

    # Add clusters to DataFrame
    df['cluster'] = clusters

    #Print number of unique clusters and the number of outlier clusters
    print(f"Number of unique clusters found: {len(df['cluster'].unique())}")
    print(f"Number of outlier reviews {len(df[df['cluster'] == -1])}")

    # Print some reviews from each cluster
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            print("Outlier reviews: Skipping")
            continue
        print(f"\n--- Cluster: {cluster_id} ---")
        cluster_df = df[df['cluster'] == cluster_id].head() #Show only top 5 reviews for each cluster
        print(cluster_df[['Reviews','cluster']])

    return df

# Example Usage
if __name__ == "__main__":
    file_path = 'AvtarReviews_csv.csv'  # Replace with the actual path to your CSV file
    clustered_df = main_clustering(file_path)

    # You can save the dataframe to CSV file here if needed
    # clustered_df.to_csv('clustered_reviews.csv', index = False)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 1. Load Data and Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'Reviews' not in df.columns:
        raise ValueError("The CSV must have a column named 'Reviews'")
    df['cleaned_review'] = df['Reviews'].apply(clean_text)
    df.drop_duplicates(subset='cleaned_review', inplace=True)
    return df

# 2. Text Embedding
def get_sentence_embeddings(texts):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)
    return embeddings

# Alternative if sentence transformer doesn't work
def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()

# 3. Clustering
def cluster_reviews(embeddings, min_cluster_size=50):
    cluster = HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = cluster.fit_predict(embeddings)
    return clusters

# 4. Select Representative Reviews
def select_representative_reviews(cluster_reviews_df, embeddings, tfidf_vectorizer, num_reviews_to_select=5):
    """Selects representative reviews based on a combination of length, similarity and feature scores."""

    if len(cluster_reviews_df) == 0:
        return []

    if len(cluster_reviews_df) <= num_reviews_to_select:
        return cluster_reviews_df["cleaned_review"].tolist()

    # Calculate review lengths
    review_lengths = cluster_reviews_df["cleaned_review"].apply(lambda x: len(x.split()))

    # Calculate similarities of reviews to each other using their embeddings
    similarities = cosine_similarity(embeddings)
    # Calculate average similarities of each review to every other review
    avg_similarities = np.mean(similarities, axis=1)

    # Calculate distances to cluster centroid
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Use TF-IDF to calculate feature scores
    tfidf_matrix = tfidf_vectorizer.transform(cluster_reviews_df["cleaned_review"].tolist())
    feature_counts = np.sum(tfidf_matrix, axis=1)


    #create a dataframe of the original reviews and their scores
    score_df = pd.DataFrame({'cleaned_review': cluster_reviews_df['cleaned_review'], 'avg_similarity': avg_similarities, 'length': review_lengths, 'centroid_distance': distances, 'feature_counts': feature_counts})

    # Normalize the scores
    max_length = score_df['length'].max()
    min_length = score_df['length'].min()
    normalized_length = score_df['length'].apply(lambda x: (x - min_length) / (max_length - min_length))

    max_similarity = score_df['avg_similarity'].max()
    min_similarity = score_df['avg_similarity'].min()
    normalized_similarity = score_df['avg_similarity'].apply(lambda x: 1-((x - min_similarity) / (max_similarity - min_similarity)))

    max_distance = score_df['centroid_distance'].max()
    min_distance = score_df['centroid_distance'].min()
    normalized_distance = score_df['centroid_distance'].apply(lambda x: (x - min_distance) / (max_distance - min_distance))

    max_feature = score_df['feature_counts'].max()
    min_feature = score_df['feature_counts'].min()
    normalized_feature = score_df['feature_counts'].apply(lambda x: (x - min_feature) / (max_feature - min_feature))

    # Calculate an overall score combining all metrics (adjust weights as needed)
    score_df['overall_score'] = 0.4*normalized_length + 0.3*normalized_similarity + 0.1*normalized_distance + 0.2*normalized_feature

    # Sort and take the top k reviews
    selected_reviews = score_df.nlargest(num_reviews_to_select, 'overall_score')['cleaned_review'].tolist()
    return selected_reviews

# --- Main Clustering Function ---
def main_clustering(file_path, num_reviews_to_select=5):
    # 1. Load and Clean Data
    try:
        df = load_and_clean_data(file_path)
    except FileNotFoundError:
        print("File not found, please verify file path.")
        return
    except ValueError as e:
        print(e)
        return

    # 2. Text Embedding
    embeddings = get_sentence_embeddings(df['cleaned_review'].tolist())
    # If sentence transformer fails try:
    # embeddings = get_tfidf_embeddings(df['cleaned_review'].tolist())

    # 3. Clustering
    clusters = cluster_reviews(embeddings, min_cluster_size=50)

    # Add clusters to DataFrame
    df['cluster'] = clusters

    # Print number of unique clusters and the number of outlier clusters
    print(f"Number of unique clusters found: {len(df['cluster'].unique())}")
    print(f"Number of outlier reviews {len(df[df['cluster'] == -1])}")

    # TF-IDF vectorizer for the whole dataset
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(df["cleaned_review"])

    all_selected_reviews = []

    # Process each cluster
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
             print("Outlier reviews: Skipping")
             continue

        print(f"\n--- Processing Cluster: {cluster_id} ---")
        cluster_df = df[df['cluster'] == cluster_id]

        cluster_size = len(cluster_df)
        print(f"Cluster {cluster_id} size: {cluster_size}")
        # Get embeddings specific to the current cluster
        cluster_embeddings = embeddings[df['cluster'] == cluster_id]

        # 4. Select Representative Reviews
        print("Selecting Representative Reviews")
        selected_reviews = select_representative_reviews(cluster_df, cluster_embeddings, tfidf_vectorizer, num_reviews_to_select=num_reviews_to_select)
        print(f"Selected Reviews: {selected_reviews}")

        all_selected_reviews.extend(selected_reviews) #Add all selected reviews to a list

    return df, all_selected_reviews #return selected reviews for every cluster

# Example Usage
if __name__ == "__main__":
    file_path = 'AvtarReviews_csv.csv'  # Replace with the actual path to your CSV file
    num_reviews_to_select_per_cluster = 10
    clustered_df, selected_reviews = main_clustering(file_path, num_reviews_to_select= num_reviews_to_select_per_cluster)

    print("\n--- All Selected Reviews ---")
    for review in selected_reviews:
        print(review)

     # You can save the dataframe to CSV file here if needed
    # clustered_df.to_csv('clustered_reviews.csv', index = False)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 1. Load Data and Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'Reviews' not in df.columns:
        raise ValueError("The CSV must have a column named 'Reviews'")
    df['cleaned_review'] = df['Reviews'].apply(clean_text)
    df.drop_duplicates(subset='cleaned_review', inplace=True)
    return df

# 2. Text Embedding
def get_sentence_embeddings(texts):
  model = SentenceTransformer('all-mpnet-base-v2')
  embeddings = model.encode(texts)
  return embeddings

# Alternative if sentence transformer doesn't work
def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()

# 3. Clustering (K-Means) - modified to find optimal clusters and calculate silhouette score
def cluster_reviews(embeddings, max_clusters = 10, random_state = 42):
    """Clusters the embeddings using k-means and selects optimal k based on silhouette score"""

    silhouette_scores = []
    clusters_range = range(2, max_clusters)

    for k in clusters_range:
        kmeans = KMeans(n_clusters = k, random_state= random_state, n_init = 'auto')
        clusters = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, clusters)
        silhouette_scores.append(score)

    #get the value of k that has the highest silhouette score.
    best_k = clusters_range[np.argmax(silhouette_scores)]
    print(f"Optimal k using silhouette scores: {best_k}")

    #run k-means again with the optimal k
    kmeans = KMeans(n_clusters = best_k, random_state= random_state, n_init = 'auto')
    clusters = kmeans.fit_predict(embeddings)
    return clusters


# --- Main Clustering Function ---
def main_clustering(file_path):
    # 1. Load and Clean Data
    try:
        df = load_and_clean_data(file_path)
    except FileNotFoundError:
        print("File not found, please verify file path.")
        return
    except ValueError as e:
        print(e)
        return

    # 2. Text Embedding
    embeddings = get_sentence_embeddings(df['cleaned_review'].tolist())
    # If sentence transformer fails try:
    # embeddings = get_tfidf_embeddings(df['cleaned_review'].tolist())

    # 3. Clustering using k-means
    clusters = cluster_reviews(embeddings)

    # Add clusters to DataFrame
    df['cluster'] = clusters

    # Print cluster sizes
    cluster_sizes = df['cluster'].value_counts().sort_index()
    print("\n--- Cluster Sizes ---")
    for cluster_id, size in cluster_sizes.items():
        print(f"Cluster {cluster_id}: {size} reviews")

    return df

# Example Usage
if __name__ == "__main__":
    file_path = 'AvtarReviews_csv.csv'  # Replace with the actual path to your CSV file
    clustered_df = main_clustering(file_path)

    if clustered_df is not None:
        print(clustered_df.head())

        # You can save the dataframe to CSV file here if needed
        clustered_df.to_csv('clustered_reviews.csv', index=False)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 1. Load Data and Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'Reviews' not in df.columns:
        raise ValueError("The CSV must have a column named 'Reviews'")
    df['cleaned_review'] = df['Reviews'].apply(clean_text)
    df.drop_duplicates(subset='cleaned_review', inplace=True)
    return df

# 2. Text Embedding
def get_sentence_embeddings(texts):
  model = SentenceTransformer('all-mpnet-base-v2')
  embeddings = model.encode(texts)
  return embeddings

# Alternative if sentence transformer doesn't work
def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()

# 3. Clustering (K-Means)
def cluster_reviews(embeddings, max_clusters = 10, random_state = 42):
    """Clusters the embeddings using k-means and selects optimal k based on silhouette score"""

    silhouette_scores = []
    clusters_range = range(2, max_clusters)

    for k in clusters_range:
        kmeans = KMeans(n_clusters = k, random_state= random_state, n_init = 'auto')
        clusters = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, clusters)
        silhouette_scores.append(score)

    #get the value of k that has the highest silhouette score.
    best_k = clusters_range[np.argmax(silhouette_scores)]
    print(f"Optimal k using silhouette scores: {best_k}")

    #run k-means again with the optimal k
    kmeans = KMeans(n_clusters = best_k, random_state= random_state, n_init = 'auto')
    clusters = kmeans.fit_predict(embeddings)
    return clusters

# 4. Select Useful Reviews Method 3 (Combined Length and Diversity)
def select_useful_reviews_length_and_diversity(cluster_df, embeddings, k = 2500, weight_length = 0.5, weight_diversity = 0.5):
    """Select top k reviews based on length and diversity"""
    if len(cluster_df) == 0:
        return []

    if len(cluster_df) <= k:
        return cluster_df["cleaned_review"].tolist()

    # Calculate review lengths
    review_lengths = cluster_df["cleaned_review"].apply(lambda x: len(x.split()))

    # Calculate similarities of reviews to each other using their embeddings
    similarities = cosine_similarity(embeddings)
    # Calculate average similarities of each review to every other review
    avg_similarities = np.mean(similarities, axis=1)

    #create a dataframe of the original reviews and their scores
    score_df = pd.DataFrame({'cleaned_review': cluster_df['cleaned_review'], 'avg_similarity': avg_similarities, 'length': review_lengths})

    # Normalize the scores
    max_length = score_df['length'].max()
    min_length = score_df['length'].min()

    # For length the bigger the better
    normalized_length = score_df['length'].apply(lambda x: (x - min_length) / (max_length - min_length))
    # For average similarity the smaller the better
    max_similarity = score_df['avg_similarity'].max()
    min_similarity = score_df['avg_similarity'].min()
    normalized_similarity = score_df['avg_similarity'].apply(lambda x: 1-((x - min_similarity) / (max_similarity - min_similarity)))
    # calculate an overall score which combines both normalized length and avg_similarity (50-50 weightage)
    score_df['overall_score'] = weight_length*normalized_length + weight_diversity*normalized_similarity

    #sort the reviews and take the top k
    selected_reviews = score_df.nlargest(k,'overall_score')['cleaned_review'].tolist()
    return selected_reviews


# --- Main Clustering Function ---
def main_clustering(file_path, k = 2500, weight_length = 0.5, weight_diversity = 0.5):
   # 1. Load and Clean Data
    try:
        df = load_and_clean_data(file_path)
    except FileNotFoundError:
        print("File not found, please verify file path.")
        return
    except ValueError as e:
        print(e)
        return

    # 2. Text Embedding
    embeddings = get_sentence_embeddings(df['cleaned_review'].tolist())
    # If sentence transformer fails try:
    # embeddings = get_tfidf_embeddings(df['cleaned_review'].tolist())

    # 3. Clustering
    clusters = cluster_reviews(embeddings)

    # Add clusters to DataFrame
    df['cluster'] = clusters

    # Print cluster sizes
    cluster_sizes = df['cluster'].value_counts().sort_index()
    print("\n--- Cluster Sizes ---")
    for cluster_id, size in cluster_sizes.items():
        print(f"Cluster {cluster_id}: {size} reviews")

    # Save Initial Clusters to CSV
    initial_clusters_file_path = "initial_clusters.csv"
    df.to_csv(initial_clusters_file_path, index=False)
    print(f"Initial clusters saved to: {initial_clusters_file_path}")

    all_selected_reviews_df = pd.DataFrame()

    # Process each cluster and select useful reviews
    for cluster_id in df['cluster'].unique():
        print(f"Processing cluster: {cluster_id}")
        cluster_df = df[df['cluster'] == cluster_id]

        cluster_embeddings = embeddings[df['cluster'] == cluster_id] #Get the embeddings of the current cluster
        # 4. Select Useful Reviews Method 3
        selected_reviews = select_useful_reviews_length_and_diversity(cluster_df, cluster_embeddings, k = k, weight_length = weight_length, weight_diversity = weight_diversity)
        selected_reviews_df = cluster_df[cluster_df['cleaned_review'].isin(selected_reviews)]


        if not selected_reviews_df.empty: #skip processing if no reviews
           selected_reviews_df["cluster"] = cluster_id #Adding the cluster number to data frame
           all_selected_reviews_df = pd.concat([all_selected_reviews_df, selected_reviews_df], ignore_index = True) #combine results of all clusters
        else:
           print("no useful reviews")


    return all_selected_reviews_df

# Example Usage
if __name__ == "__main__":
    file_path = 'Remente.csv'  # Replace with the actual path to your CSV file
    k_reviews = 100 #Number of top reviews to select for every cluster
    weight_length = 0.7
    weight_diversity = 0.3

    selected_reviews_df = main_clustering(file_path, k = k_reviews, weight_length = weight_length, weight_diversity = weight_diversity)

    if not selected_reviews_df.empty:
        output_file_path = "Remente_selected_reviews.csv"
        selected_reviews_df.to_csv(output_file_path, index=False)
        print(f"\nSelected reviews with their cluster assignments saved to: {output_file_path}")
    else:
        print("No reviews found after filtering")

"""Take 20-25 reviews at a time. extract features and emotions

Training BART to classify requirements into FR and NFR
"""

import pandas as pd
import numpy as np
import torch
from transformers import BartTokenizerFast, BartForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report  # Added classification_report


# 1. Load and Prepare Data:
data = pd.read_csv("promise.csv", sep=",", quotechar="'", escapechar="\\", encoding='latin1', on_bad_lines='skip') # Updated to handle your file format
data = data[["Requirement", "Type"]] # Select the relevant columns
data = data.dropna() # Remove rows with missing values


# Convert labels to numerical representation (0 for F, 1 for NFR)
data['Type'] = data['Type'].map({'F': 0, 'NFR': 1})

# Split into train and test sets (stratified to maintain class proportions)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['Requirement'], data['Type'], test_size=0.2, random_state=42, stratify=data['Type']
)
# Resetting the index of train_labels and test_labels to avoid KeyError
train_labels = train_labels.reset_index(drop=True)  # Reset index and drop old index
test_labels = test_labels.reset_index(drop=True)  # Reset index and drop old index

# 2. Tokenization:
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)


# 3. Create a Dataset Object:
class RequirementsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = RequirementsDataset(train_encodings, train_labels)
test_dataset = RequirementsDataset(test_encodings, test_labels)



# 4. Model Initialization (same as before)
model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=2)


# 5. Training Arguments (modified):
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # Increased epochs (adjust as needed)
    per_device_train_batch_size=4,  # Reduced batch size (adjust as needed)
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,               # Increased logging steps
    eval_strategy="epoch",
    save_strategy="epoch",          # Save every epoch
    load_best_model_at_end=True,
    metric_for_best_model="f1",       # Metric for best model selection
    save_total_limit=2,           # Save only the best 2 checkpoints to avoid filling up disk space
)





# 6. Define Metrics (F1-score - same as before)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')  # Use weighted F1 for imbalanced classes
    acc = accuracy_score(labels, preds)
    return {
        'f1': f1,
        'accuracy': acc,
    }



# 7. Trainer:
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # function to compute metrics
)



# 8. Fine-tune the model:
trainer.train()



# 9. Evaluation (improved):
print("Evaluating on test set:")
results = trainer.evaluate(test_dataset)  # Get detailed results
print(results)


predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions[0], axis=1)
print(classification_report(test_labels, predicted_labels))  # Print classification report




# 10. Save the fine-tuned model (same as before)
trainer.save_model("./best_bart_model")



# 11. Prediction on new data:
def predict_requirement_type(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = np.argmax(outputs.logits.detach().numpy())
    return "functional" if predicted_class == 0 else "non-functional"

example_requirement = "The system shall allow users to log in."
predicted_type = predict_requirement_type(example_requirement)
print(f"Requirement: {example_requirement}")
print(f"Predicted Type: {predicted_type}")

count=0
try:
    results = []
    with open('requirements.csv', 'r') as file:
        for line in file:
            #print(line)
            predicted_type = predict_requirement_type(line)
            if predicted_type == "functional":
                print(f"Requirement: {line}") #Indented these lines
                print(f"Predicted Type: {predicted_type}") #Indented these lines
                count+=1 #Indented these lines
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
print(count)

"""Mapping Feature to quality aspect"""

!pip install -q -U langchain langchain_core together
!pip install langchain-together
from google.colab import userdata
!pip install -q -U langchain langchain_core
from langchain_together.llms import Together

from langchain_core.prompts import ChatPromptTemplate

rag_template = """\
You are AI assistant for requirements analysis. You will help to identify dependencies among the following functional requirements and the specified categories.

The functional requirements are:

1. Allow users (students, elderly, caregivers, teachers) to create and manage profiles.
2. Enable role-based access (e.g., student, teacher, caregiver).
3. Support personalized well-being recommendations based on user data.
4. Integrate IoT sensors and wearables to collect health and environmental data (e.g., heart rate, sleep, air quality).
5. Provide real-time health monitoring and alerts.
6. Track mental and emotional well-being through self-assessment tools.
7. Offer personalized physical activity suggestions based on user needs.
8. Provide dietary recommendations based on health conditions.
9. Suggest mindfulness and stress-relief exercises.
10. Send real-time alerts for abnormal health readings.
11. Notify caregivers or teachers about emergencies.
12. Provide reminders for medication, hydration, and physical activities.
13. Adjust smart home or school environment settings based on user comfort (e.g., lighting, temperature, noise levels).
14. Provide real-time feedback on environmental conditions affecting well-being.
15. Enable peer support and communication between users (e.g., forums, chat).
16. Facilitate teacher-student or caregiver-elderly interaction.
17. Provide group activities or challenges to promote social engagement.
18. Generate well-being reports based on collected data.
19. Provide predictive analytics for health trends.
20. Allow caregivers and professionals to analyze user data for better decision-making.
21. Support interoperability with healthcare systems, wearables, and IoT platforms.
22. Enable data sharing with healthcare professionals securely.
23. Implement GDPR-compliant data protection measures.
24. Provide users with control over data sharing and privacy settings.
25. Encrypt sensitive well-being data.
26. Include reward systems to encourage healthy habits.
27. Provide progress tracking and achievement badges.
28. Offer interactive challenges for engagement.




Analyze these requirements from the perspective of an expert requirements engineer. Based on your understanding of different types of requirements and potential interaction points, you will be looking for dependencies (positive or negative dependency) across the following categories:

1. Usability- Requirements aimed at ensuring the application is intuitive, efficient, and engaging for diverse users.
2. Safety- Ensures user protection from harm through secure data handling and safe system interactions.
3. Cultural and Regional Sensitivity- Adapts to diverse cultural values, languages, and regional health practices.
4. Aesthetics- The visual and interactive appeal of the system, influencing user experience through design, layout, and overall look and feel.
5. Reliability- Ensures consistent performance, minimal downtime, and accurate data processing.
6. Personalizable – Allows users to customize features based on preferences, needs, and accessibility.



Map the functional requirements to one or more of these categories. Mark which dependency are positive and which are negative. If a functional requirement cannot be mapped to any of the above categories, do not provide a false mapping. Here are some examples for your reference
Requirement: The system must provide captions for all video-based lectures.

Category: Usability
Requirement- The application shall provide an intuitive dashboard with easy navigation, tooltips, and onboarding tutorials for first-time users.

Category: Safety
Requirement- The system shall issue emergency alerts to caregivers or medical professionals when abnormal health parameters are detected.

Category: Cultural and Regional Sensitivity
Requirement– The application shall support multiple languages and allow users to select region-specific health guidelines.

Category: Aesthetics
Requirement – The interface shall use a visually appealing design with a customizable theme, ensuring readability and accessibility.

Category: Reliability
Requirement – The system shall store and sync user data in real-time, ensuring no data loss during network disruptions.

Category: Personalizable
Requirement– The application shall allow users to set personal goals, adjust notification preferences, and modify dashboard widgets.

"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

from together import Together

client = Together(api_key='.........')

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": rag_template}],
)

print(response.choices[0].message.content)

!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
input_csv = 'WellBeingCombined.csv'
output_csv = 'WellBeing_feature_emotion.csv'
with open(input_csv, 'r', encoding='latin-1') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile: # Changed encoding to latin-1 for infile and utf-8 for outfile
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["Review", "Feature","Emotion"])  # Write header row

        # Skip header row if it exists
        #next(reader, None)

        for row in reader:
            review = row['Review']
            feature= row['Feature']
            analyzer = SentimentIntensityAnalyzer()
            vs = analyzer.polarity_scores(review)
            if vs['compound'] >= 0.05 :
              emotion="Positive"
            elif vs['compound'] <= - 0.05 :
              emotion="Negative"
            else :
              emotion="Both Positive and Negative"
            writer.writerow([review, feature, emotion])
            #print(f"Processed query: {user_query}")

!pip install langchain-huggingface
!pip install -U langchain-community
!pip install chromadb
import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = 'WellBeingKnowledge.csv'
data = pd.read_csv(file_path)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings()

# Compute semantic similarity using TF-IDF for grouping rows based on 'Feature'
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Feature'])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Group rows based on similarity
visited = set()
grouped_chunks = []

for i in range(len(data)):
    if i in visited:
        continue
    # Find similar rows for the current row
    similar_indices = [j for j in range(len(data)) if similarity_matrix[i][j] > 0.5 and j not in visited]
    # Limit each group to a maximum of 5 rows
    chunk_indices = similar_indices[:5]
    visited.update(chunk_indices)

    # Combine rows into a single chunk
    chunk_text = "\n---\n".join(
        [f"Review: {data.iloc[j]['Review']}\nFeature: {data.iloc[j]['Feature']}\nEmotion: {data.iloc[j]['Emotion']}" for j in chunk_indices]
    )
    grouped_chunks.append(Document(page_content=chunk_text))

# Initialize Chroma database
persist_directory = "./chroma_db"
vectordb = Chroma.from_documents(documents=grouped_chunks,
                                 embedding=embedding_model,
                                 persist_directory=persist_directory)

# Persist the database for later use
vectordb.persist()

print("Data successfully stored in Chroma with semantic grouping based on features!")

semantic_chunk_retriever = vectordb.as_retriever(search_kwargs={"k" : 5})

def retrieve_relevant_docs(user_query):
  retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})#search_type="similar"
  results = retriever.invoke(user_query)
  searched_content = []
  for result in results:
      #print(result.page_content)
      searched_content.append(result.page_content)

  combined_content = " ".join(searched_content)

  # Print or use the combined string
  # print(combined_content)
  return combined_content

from langchain_core.prompts import ChatPromptTemplate
import openai

rag_prompt = """\

You are AI assistant for requirements analysis.Your task is to predict the likely emotional responses of different user demographics to a given functional requirement for a well being software system.}


   Carefully consider the provided functional requirement as user query. Understand its purpose, functionality, and potential impact on users.  The FR will be provided in the following format:}


Evaluate the FR from the perspective of the following user demographics.  Consider their specific needs, goals, technical expertise, and potential frustrations.}


 Demographic 1: Individuals with technical background
 Demographic 2: Novice Individuals from non technical background


Access the provided contexts to find relevant examples of past user feedback, reviews, and emotional responses to similar features of a different healthcare application system.


Generate the emotional response with respect to different user demographic into one or more of the following category of emotions-


1. Empowered
2. Hopeful
3. Ashamed
4. Feel respected
5. Engaging
6. Curiosity
7. Frustated
8. Connected


"""
#Demographic 3: Healthcare Workers (Doctor, Nurse, Lab technician) with technical background
#  Demographic 4: Novice Healthcare Workers (Doctor, Nurse, Lab technician) from non technical background


from together import Together
client = Together(api_key='...........')
import csv

sleep_time= 20
input_csv_path = 'WellBeingRequirement.csv'
output_csv_path = 'WellBeing_identified_emotion_mistral.csv'
with open(input_csv_path, 'r', encoding='latin-1') as infile, open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile: # Changed encoding to latin-1 for infile and utf-8 for outfile
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["Requirement", "Response"])  # Write header row

    for row in reader:
        # Access the column using its actual key 'ï»¿Requirements' or any of its aliases
        # Replace 'ï»¿Requirements' with the actual key if it's different
        user_query = row.get('ï»¿Requirement', row.get('Requirement', None))

        # If the key is truly missing, skip the row
        if user_query is None:
            print(f"Skipping row due to missing key: {row}")
            continue

        # The rest of your code remains unchanged
        combined_content = retrieve_relevant_docs(user_query)

        completion = client.chat.completions.create(
            #model="hf:meta-llama/Meta-Llama-3.1-405B-Instruct",
            model="mistralai/Mixtral-8x22B-Instruct-v0.1",
            #model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Based on the given context answer the user query "},
                {"role": "assistant", "content": f"Here is some context: {combined_content}"},
                {"role": "user", "content": f"Here is the {user_query + rag_prompt}"}
            ]
        )

        response = completion.choices[0].message.content
        writer.writerow([user_query, response])
        #time.sleep(sleep_time)
        print(f"Processed query: {user_query}")

client = Together(api_key='.......')

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    messages=[{"role": "user", "content": rag_template}],
)