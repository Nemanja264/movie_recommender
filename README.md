
# Movie Recommendation System

This is a **movie recommendation app** built using **Flask**, **machine learning**, and **MongoDB**. The app provides movie recommendations based on a movie title input and the number of recommendations requested. It uses a **fine-tuned machine learning model** to recommend similar movies and displays them along with their IMDb links.

## Features

- **Real-time Movie Recommendations**: Get personalized movie recommendations based on a movie title.
- **Fine-tuned Model**: A machine learning model fine-tuned on movie data to generate recommendations.
- **Frontend**: Interactive UI using **JavaScript** for handling user input and displaying results.
- **Backend**: Built with **Flask**, serving recommendations and handling API requests.
- **Database**: MongoDB is used to store similarity matrices and movie data.

## Prerequisites

Before running the app, make sure you have the following:

- Python 3.x
- Flask
- MongoDB (locally or via a cloud service like MongoDB Atlas)

### **Required Libraries**

You can install all the necessary Python packages by running:

```bash
pip install -r requirements.txt
```

### **Install Dependencies**
Ensure you have the following dependencies in your **`requirements.txt`** file:

```
Flask==2.0.1
pandas==1.3.0
scipy==1.7.0
sentence-transformers==2.1.0
sklearn==0.0
pymongo==3.12.0
requests==2.26.0
```

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

## Setup and Running the Application

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

## Fine-Tuned Model

The app uses a **fine-tuned model** located in the `movie_rec200` folder. This model was trained on movie descriptions to generate accurate recommendations. 

- The fine-tuned model was created using **Sentence Transformers**.
- It processes movie titles, overviews, genres, and other metadata to find similar movies.

## Contributing

Feel free to fork the repository, open issues, and make pull requests. If you have any suggestions or improvements, we welcome your contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Flask**: Web framework used for backend development.
- **Sentence Transformers**: Used for generating movie embeddings and fine-tuning the recommendation model.
- **MongoDB**: Used for storing and managing movie data and similarity matrices.
- **Movie Database (TMDB)**: For retrieving movie data (if integrated).
