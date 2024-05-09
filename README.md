# IMDB Review Sentiment Analysis 🎬

## Overview 📖
This project leverages advanced machine learning and natural language processing techniques to perform sentiment analysis on a dataset of 50,000 IMDB movie reviews. The objective is to classify each review as positive or negative, providing insights into the general sentiment expressed by the movie-going audience.

## Data 📊
The IMDB dataset, split into 25,000 training and 25,000 testing reviews, is used for this analysis. Each review is labeled as either positive or negative, representing the sentiment. The dataset offers a balanced approach to sentiment analysis, with an equal number of positive and negative reviews.

## Natural Language Processing (NLP) 🧠
The project utilizes various NLP techniques to preprocess and transform textual data into a format suitable for machine learning:
- **Text Cleaning**: HTML tags and special characters are removed to simplify the texts.
- **Tokenization**: Reviews are split into individual words or tokens.
- **Stopword Removal**: Common words that add little value to sentiment analysis are filtered out.
- **TF-IDF Vectorization**: Transforms text into a meaningful representation of numbers which is used to fit machine learning algorithms.

## Machine Learning Models 🤖
Two primary machine learning models are trained to classify the sentiments of the reviews:
- **Logistic Regression**: This model serves as the baseline for our analysis. It is known for its efficiency and effectiveness in binary classification tasks.
- **Random Forest Classifier**: Known for its high accuracy, this ensemble model uses multiple decision trees to improve the predictive performance and control over-fitting.

## Model Training and Evaluation 📈
Models are trained on the preprocessed data, evaluating their performance through:
- **Accuracy**: Measures the overall effectiveness of the model.
- **Precision, Recall, and F1-Score**: These metrics help in understanding the balance between the precision and the recall for our models.

## Results and Insights 📝
- **Model Performance**: The Logistic Regression model achieved an accuracy of 87.72%, while the Random Forest model showed an accuracy of 83.37%.
- **Comparative Analysis**: The insights from the models help in understanding different aspects of model performance and guide further refinement.

## Visualizations 🌟
Word Clouds are generated to visualize the most frequent words found in positive and negative reviews, illustrating the common themes and terms used in different sentiments.

## Saving Models 💾
Models are saved using `joblib` for reuse without retraining:
```python
import joblib
joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
