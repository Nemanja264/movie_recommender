
# Movie Recommendation System

This is a **movie recommendation app** built using **Flask**, **machine learning**, and **MongoDB**. The app provides movie recommendations based on a movie title input and the number of recommendations requested. It uses a **fine-tuned machine learning model** to recommend similar movies and displays them along with their IMDb links.

## Features

- **Real-time Movie Recommendations**: Get personalized movie recommendations based on a movie title.
- **Fine-tuned Model**: A machine learning model fine-tuned on movie data to generate recommendations.
- **Frontend**: Interactive UI using **JavaScript** for handling user input and displaying results.
- **Backend**: Built with **Flask**, serving recommendations and handling API requests.
- **Database**: MongoDB is used to store similarity matrices and movie data.

## Fine-Tuned Model

The app uses a **fine-tuned model** for generating movie recommendations, which was trained using **Sentence Transformers**.

### **Fine-Tuning Process**:

1. **Data Preparation**:
   - The movie data is processed to include **title**, **genre**, **release year**, and **overview**. These features are combined into a **'combined'** column, which is used as input for the model.
   
2. **Model Selection**:
   - The **`SentenceTransformer`** model (`all-MiniLM-L6-v2`) is used for generating dense vector representations of the movie's text features.

3. **Similarity Calculation**:
   - A custom similarity function (`get_overall_similarity`) was developed to calculate the similarity between two movies using their **title**, **genre**, **release year**, and **overview**. Weights are assigned to each feature based on their importance in the recommendation process.

4. **Training**:
   - The model was fine-tuned using **CosineSimilarityLoss**, training on pairs of movies and their calculated similarity score. The model was trained on the **40 movies** in the dataset, and the fine-tuned model was saved as `movie_rec2`.

5. **Model Saving**:
   - The fine-tuned model is saved in the directory `movie_rec200/` for later use in making movie recommendations.

6. **Usage**:
   - The fine-tuned model is loaded during the app's initialization and used to compute the similarity scores for generating recommendations.

## Live Application

You can access the live application here: [https://movique-u71e.onrender.com](https://movique-u71e.onrender.com)

## Acknowledgements

- **Flask**: Web framework used for backend development.
- **Sentence Transformers**: Used for generating movie embeddings and fine-tuning the recommendation model.
- **MongoDB**: Used for storing and managing movie data and similarity matrices.
- **Movie Database (TMDB)**: For retrieving movie data (if integrated).
