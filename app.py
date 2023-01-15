from flask import Flask, request
import model
import imp
imp.reload(module=model)

app = Flask(__name__)

print("Loading data set")
jobs = model.load_dataset()
print("Preprocessing data set")
jobs_base_line = model.pre_process_dataset(jobs)
print("Training model")
indices, cosine_sim, titles = model.train_model(jobs_base_line)


@app.route('/recommend_job', methods=['GET'])
def recommend_job():
    title = request.args.get('title')
    return model.get_recommendations(title, indices, cosine_sim, titles)


#'SAP Business Analyst / WM'
