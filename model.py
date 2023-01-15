import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset():
    jobs = pd.read_csv('dataset/jobs.csv', delimiter=',',
                       encoding='utf-8', error_bad_lines=False)
    return jobs


def pre_process_dataset(jobs):
    jobs_base_line = jobs.iloc[0:1000, 0:8]
    jobs_base_line['title'] = jobs_base_line['title'].fillna('')
    jobs_base_line['skillsets'] = jobs_base_line['skillsets'].fillna('')
    jobs_base_line['skillsets'] = jobs_base_line['title'] + \
        jobs_base_line['skillsets']
    return jobs_base_line


def train_model(jobs_US_base_line):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                         min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(jobs_US_base_line['skillsets'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    jobs_US_base_line = jobs_US_base_line.reset_index()
    titles = jobs_US_base_line['title']
    indices = pd.Series(jobs_US_base_line.index, index=jobs_US_base_line['title'])
    return indices, cosine_sim, titles

def get_recommendations(title, indices, cosine_sim, titles):
    idx = indices[title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    job_indices = [i[0] for i in sim_scores]
    return (titles.iloc[job_indices]).head(10).to_json()
