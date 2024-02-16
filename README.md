# Job Title Recommendation System

This project is a recommendation system for job titles based on their skillsets. It uses a dataset of job titles and their corresponding skillsets to train a model that can recommend similar job titles based on a given input. The project is built using Python and the Flask web framework.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/job-title-recommendation-system.git
   ```

2. Run the Flask web application:

   ```bash
   python app.py
   ```

4. Open your web browser and navigate to `http://localhost:5000/recommend_job?title=your-job-title` to get recommendations for a specific job title.

## Project Structure

The project is structured as follows:

- `app.py`: The main Flask application that serves the recommendation API.
- `model.py`: Contains functions for loading the dataset, preprocessing the data, training the model, and getting recommendations.
- `dataset/jobs.csv`: The dataset of job titles and their skillsets.
