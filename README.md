# Intelligent-Book-Recommendation-Web-App📚🔮
# Book Recommendation System 📚🔮

A comprehensive web-based system that provides personalized book recommendations using advanced machine learning techniques. The system is designed to enhance the book discovery experience by suggesting books based on user preferences and similar reading patterns.

## Key Features 🚀

### 1. **Data Analysis Pipeline 🧺🔧**
- **Data Preprocessing and Cleaning**: The system ensures robust insights by preprocessing and cleaning raw data before use.
- **Exploratory Data Analysis (EDA)**: Interactive visualizations and detailed analysis are provided to explore the data.

### 2. **Collaborative Filtering 🔄**
- **Memory-based Techniques**:
  - **User-based Recommendations**: Recommends books based on similar users' reading preferences.
  - **Item-based Recommendations**: Suggests books that are similar to those the user has liked in the past.

### 3. **Supervised & Unsupervised Machine Learning Models 🤖**
- Implements machine learning algorithms to improve the recommendation system over time.
- **Performance Evaluation**: Includes accuracy metrics and Mean Squared Error (MSE) analysis to assess prediction precision.

### 4. **Genre Fetching Using Wikipedia API 🎨**
- Automatically fetches the genre of books that were missing this data from the dataset using the **Wikipedia API**. 
- Ensures that all books have accurate genre information for better recommendations and more insightful book discovery.

### 5. **Web Application (HTML, CSS, JavaScript, Flask) 🌐**
- A user-friendly front-end built with HTML, CSS, Bootstrap, and JavaScript.
- **Python Flask** is used to serve the backend and integrate the machine learning models.

### 6. **User Authentication (SQLite & Google API Login) 🔒**
- **SQLite Database**: Stores user login and signup data securely.
- **Google API Login**: Users can log in using their Google accounts for convenience and security.

## Technologies Used 💡

- **Python**: The backbone of the recommendation engine, used for data analysis and machine learning.
- **Pandas**: For data manipulation and cleaning.
- **Scikit-learn**: For implementing machine learning algorithms (Collaborative Filtering, etc.).
- **Matplotlib & Seaborn**: For data visualization and EDA.
- **Wikipedia API**: To fetch missing genre information for books.
- **SQLite**: For storing user authentication data (login/signup).
- **Google API**: For Google OAuth login integration.
- **Flask**: For backend development.
- **HTML/CSS/JavaScript**: For the front-end of the web application.
- **Bootstrap**: For responsive design and UI components.

## How It Works 🔧

1. **Data Collection**: Raw data about books, users, and ratings is collected and stored.
2. **Data Cleaning**: Missing values, duplicates, and outliers are handled to ensure the quality of the data.
3. **Genre Fetching**: The Wikipedia API is used to fetch genre information for books that do not have this data in the dataset.
4. **User Authentication**: 
   - **SQLite** stores user login and signup information securely.
   - Users can log in via Google using the **Google OAuth API** for an easy authentication process.
5. **Model Training**: Machine learning models (both supervised and unsupervised) are trained using collaborative filtering techniques.
6. **Recommendation Generation**: Based on user input, the system generates personalized book recommendations.
7. **User Interface**: Users interact with the web application, where they can view book recommendations based on their preferences.

## Installation 📥

To run the Book Recommendation System locally:

1. Clone this repository:
  git clone https://github.com/your-username/book-recommendation-system.git

2. Navigate into the project directory:
  cd book-recommendation-system

3. Install the required dependencies:
  pip install -r requirements.txt

4. Set up your Google OAuth credentials:
  Go to the Google Developer Console.
  Create a new project and enable the "Google+ API".
  Create OAuth 2.0 credentials and download the JSON file.
  Place the JSON credentials file in the project directory and rename it to client_secrets.json.

5. Run the Flask application:
  python app.py

6. Open your browser and visit:
  http://127.0.0.1:5000

Performance Evaluation 📊

The system evaluates the performance using:

Accuracy Metrics: Evaluate how well the model predicts user preferences.
Mean Squared Error (MSE): Provides insights into the precision of the recommendations.

