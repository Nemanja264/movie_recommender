
# Movie Recommendation System

This is a **movie recommendation app** built using **Flask**, **machine learning**, and **MongoDB**. The app provides movie recommendations based on a movie title input and the number of recommendations requested. It uses a **fine-tuned machine learning model** to recommend similar movies and displays them along with their IMDb links.

## Project Overview

The **Movie Recommendation System** app uses machine learning to recommend similar movies based on a given movie title. The system is trained using a **fine-tuned model** based on **Sentence Transformers** that generates movie embeddings to find similarities between movies. The app also utilizes **Flask** for the backend API and **MongoDB** to store the similarity matrix and movie metadata. 

### **Technologies Used**:
- **Flask**: Web framework used to create the backend server.
- **Sentence Transformers**: Used to generate movie embeddings and fine-tune the recommendation model.
- **MongoDB**: Used to store and manage the movie data and similarity matrices.
- **JavaScript (Vanilla)**: Handles dynamic user interface interactions and communicates with the backend API.
- **HTML/CSS**: The frontend is designed using basic HTML and CSS for structure and styling.

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

## Project Structure

The project consists of the following files:

```
/movie-recommender-app
│
├── app.py                # Flask backend
├── movie_recommender.py  # Movie recommendation logic
├── db.py                 # MongoDB database interaction
├── requirements.txt      # Python dependencies
├── static/
│   └── style.css         # Stylesheet for the frontend
├── templates/
│   └── index.html        # HTML file for the frontend
├── movie_rec200/         # Fine-tuned machine learning model directory
├── movies.csv            # Movie dataset for recommendations
└── genres.json           # Genre mapping file for movie genres
```

### **File Descriptions**:
- **`app.py`**: The main **Flask** backend. Handles the web server and serves recommendations by calling the movie recommendation logic in `movie_recommender.py`.
- **`movie_recommender.py`**: Contains the **movie recommendation logic** and handles the **movie similarity calculation**, loading the fine-tuned model, and generating recommendations.
- **`db.py`**: Handles **MongoDB** interactions, such as inserting movie data, querying for similar movies, and retrieving movie metadata.
- **`requirements.txt`**: Contains all the necessary **Python dependencies** for the project.
- **`style.css`**: The **CSS** file for styling the frontend and ensuring the app has a clean, user-friendly interface.
- **`index.html`**: The **HTML file** that renders the page and contains the layout of the frontend UI.
- **`movie_rec200/`**: Folder containing the **fine-tuned machine learning model** used for generating movie recommendations.
- **`movies.csv`**: The dataset containing movie details such as **title**, **genre**, **overview**, and **release year**.
- **`genres.json`**: Mapping of **genre IDs** to genre names for movie classification.

## Installation and Setup

### 1. **Clone the Repository**

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd movie-recommender-app
```

### 2. **Install Dependencies**

Install all required libraries using the following command:

```bash
pip install -r requirements.txt
```

### 3. **MongoDB Setup**

Ensure that you have a **MongoDB instance** running. You can either use a **local MongoDB database** or **MongoDB Atlas** (cloud service).

- Update the `connection_string` variable in `db.py` with your MongoDB connection URI.

### 4. **Run the Application**

Start the Flask app by running:

```bash
python app.py
```

The app will be accessible at `http://localhost:5555` in your browser.

### 5. **Using the App**

1. Open the app in your browser.
2. Enter a movie title in the input field and specify the number of recommendations you want.
3. Click **"Show Recommendations"** to get similar movie suggestions.
4. The recommendations will be displayed along with their release years and IMDb links.

## Acknowledgements

- **Flask**: Web framework used for backend development.
- **Sentence Transformers**: Used for generating movie embeddings and fine-tuning the recommendation model.
- **MongoDB**: Used for storing and managing movie data and similarity matrices.
- **Movie Database (TMDB)**: For retrieving movie data (if integrated).
