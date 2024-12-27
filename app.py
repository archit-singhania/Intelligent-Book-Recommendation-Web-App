import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import json
import logging
import os
import functools
import utils
from flask_sqlalchemy import SQLAlchemy
from authlib.integrations.flask_client import OAuth
from io import BytesIO
from flask import Flask, flash, render_template, request, jsonify, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
from PIL import Image
from functools import wraps
from extensions import db
from models import User
from utils import (
    load_users, load_books_w_r, load_books_wo_r, load_ratings,
    collaborative_filtering, get_books_by_search, get_books_by_rating,
    analyze_books_dynamic, scrape_book_details, get_user_insights, get_ratings_analysis, 
    init_db
)

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///books.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Secret Key
app.config['SECRET_KEY'] = utils.get_secret_key()

# Initialize SQLAlchemy with the app
db.init_app(app)

# Initialize database only once
with app.app_context():
    db.create_all()
    print("Database initialized successfully.")

## OAuth configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', 'abcdxyz123'),  # Replace with correct client ID
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', 'abcdxyz123'),  # Replace with correct client Secret
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Login required decorator
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Route protection
@app.before_request
def check_login():
    """Restrict access to protected routes."""
    public_routes = ['index', 'login', 'signup', 'google_login', 'google_authorized', 'login_or_signup', 'static']
    if request.endpoint and request.endpoint not in public_routes and 'user_id' not in session:
        flash('Please login to continue.', 'warning')
        return redirect(url_for('login'))

db_initialized = False

@app.before_request
def setup_db():
    """Initialize the database once before handling any request."""
    global db_initialized
    if not db_initialized:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created.")
            db_initialized = True


@app.route('/login_or_signup')
def login_or_signup():
    """Login or Signup page"""
    logger.debug("Rendering login_or_signup page")
    return render_template('login_or_signup.html', title="Login or Signup")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User signup route"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        result = utils.create_user(name, email, password)
        if result:
            user = User.query.filter_by(email=email).first()
            session['user_id'] = user.id
            session['user_info'] = {  # Add this block
                'id': user.id,
                'name': user.name,
                'email': user.email
            }
            flash('Account created successfully! You are now logged in.', 'success')
            return redirect(url_for('index'))
        
        flash('Email already exists or registration failed.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = utils.validate_user(email, password)
        if user:
            session['user_id'] = user.id
            session['user_info'] = {  # Add this block
                'id': user.id,
                'name': user.name,
                'email': user.email
            }
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/login/google')
def google_login():
    """Initiate Google OAuth login"""
    try:
        redirect_uri = url_for('google_authorized', _external=True)
        return google.authorize_redirect(redirect_uri)
    except Exception as e:
        logger.error(f"Google login error: {e}")
        flash('Google login failed. Please try again.', 'danger')
        return redirect(url_for('login'))

@app.route('/login/google/authorized')
def google_authorized():
    """Handle Google OAuth callback"""
    try:
        token = google.authorize_access_token()
        resp = google.get('https://www.googleapis.com/oauth2/v3/userinfo')
        user_info = resp.json()

        if not user_info.get('email'):
            raise Exception("Google did not provide an email address.")

        # Check or create user in database
        user = User.query.filter_by(email=user_info['email']).first()
        if not user:
            user = User(
                name=user_info['name'],
                email=user_info['email'],
                password=None,  # No password for OAuth users
                provider='google',
                provider_user_id=user_info['sub']
            )
            db.session.add(user)
            db.session.commit()

        # Set session variables
        session['user_id'] = user.id  # Use database user ID
        session['user_info'] = {  # Add this block to match the template's expectation
            'id': user.id,
            'name': user.name,
            'email': user.email
        }
        session['logged_in'] = True

        flash("Google login successful!", "success")
        return redirect(url_for('index'))  # Redirect explicitly to 'index'

    except Exception as e:
        logger.error(f"Google authentication error: {e}")
        flash("Authentication failed. Please try again.", "danger")
        return redirect(url_for('login_or_signup'))

@app.route('/logout')
def logout():
    """User logout route"""
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('logged_in', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

# Paths to datasets
FILE_PATH = '/Users/archits/Downloads/book-recommendation-system/cleaned_datasets'
BOOKS_PATH = f'{FILE_PATH}/cleaned_books.csv'
RATINGS_PATH = f'{FILE_PATH}/cleaned_ratings.csv'
USERS_PATH = f'{FILE_PATH}/cleaned_users.csv'

# Load datasets
ratings = load_ratings(RATINGS_PATH)
books_w_r = load_books_w_r(BOOKS_PATH, ratings)
users_df = load_users(USERS_PATH)
books_wo_r = load_books_wo_r(BOOKS_PATH)
books_wo_r_df = load_books_wo_r(BOOKS_PATH)

def convert_to_native_types(obj):
    """
    Recursively convert numpy types to native Python types
    """
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

@app.route('/')
def index():
    # Load the JSON data
    data = '''
    {
        "overview": {
            "most_active_user": {
                "rating_count": 4533,
                "user_id": 278418
            },
            "most_popular_author": {
                "book_count": 636,
                "name": "Agatha Christie"
            },
            "most_rated_book": {
                "avg_rating": 10.0,
                "name": "Hatchet"
            },
            "top_book": {
                "avg_rating": 10.0,
                "name": "Hatchet"
            },
            "total_authors": 98954,
            "total_books": 271379,
            "total_users": "Data unavailable"
        },
        "visuals": {
            "chart_description": "Bar chart showing the total number of books, authors, and users.",
            "chart_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgaaGQCAYAAAByNR6YAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABDRUlEQVR4nADyBRX8iVBORw0KGgoAAAANSUhEUgAAAlgAACgrCAYAAACkKkz+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAe/klEQVR4nAD0CRT8"
        }
    }
    '''

    # Parse the JSON data
    json_data = json.loads(data)

    # Get the base64 string for the chart image
    chart_image = json_data['visuals']['chart_image']

    # Initialize img_base64 as None
    img_base64 = None
    
    try:
        # Remove the 'data:image/png;base64,' part of the string
        base64_data = chart_image.split(',')[1] if ',' in chart_image else chart_image

        # Add padding if needed to avoid 'Incorrect padding' error
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += '=' * (4 - missing_padding)

        # Decode the base64 chart image
        img_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_data))

        # Convert the image to a format that can be displayed in HTML (base64)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Error decoding image: {e}")
        img_base64 = None

    # Extract key statistics for display
    overview = json_data['overview']
    stats = {
        "most_active_user": f"ID {overview['most_active_user']['user_id']} with {overview['most_active_user']['rating_count']} ratings",
        "most_popular_author": f"{overview['most_popular_author']['name']} with {overview['most_popular_author']['book_count']} books",
        "most_rated_book": f"{overview['most_rated_book']['name']} with an average rating of {overview['most_rated_book']['avg_rating']}",
        "top_book": f"{overview['top_book']['name']} with an average rating of {overview['top_book']['avg_rating']}",
        "total_authors": overview['total_authors'],
        "total_books": overview['total_books'],
        "total_users": overview['total_users']
    }

    # Check if the user is logged in and handle accordingly
    if 'user_id' in session:
        # Add logic specific to logged-in users
        user_stats = {
            "welcome_message": f"Welcome back, User {session['user_id']}!"
        }
        return render_template('index.html', stats=stats, chart_img=img_base64, user_stats=user_stats)
    else:
        # Handle non-logged-in users
        return render_template('index.html', stats=stats, chart_img=img_base64)

def process_chart_image(chart_image):
    try:
        # Check if the string includes the base64 prefix and remove it
        if chart_image.startswith('data:image/png;base64,'):
            base64_data = chart_image.split(',')[1]
        else:
            base64_data = chart_image
        
        # Add padding if necessary
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += '=' * (4 - missing_padding)
        
        # Decode the base64 data
        img_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(img_data))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/overview')
def overview():
    # Total counts
    total_books = int(books_wo_r_df.shape[0])
    total_authors = int(books_wo_r_df['Book Author'].nunique())
    total_users = int(users_df.shape[0]) if isinstance(users_df, pd.DataFrame) else 0

    # Most Rated Book
    book_ratings = ratings.groupby('ISBN')['Rating'].agg(['count', 'mean']).reset_index()
    book_ratings = book_ratings.merge(books_wo_r_df[['ISBN', 'Book Title', 'Book Author']], on='ISBN', how='left')
    most_rated_book = book_ratings.loc[book_ratings['count'].idxmax()]
    
    # Top Rated Book
    top_rated_book = book_ratings.loc[book_ratings['mean'].idxmax()]

    # Most Active User
    user_activity = ratings.groupby('User Id').agg(
        rating_count=('Rating', 'count'),
        avg_rating=('Rating', 'mean')
    ).reset_index()
    most_active_user = user_activity.loc[user_activity['rating_count'].idxmax()]

    # Most Popular Author
    author_popularity = books_wo_r_df['Book Author'].value_counts()
    most_popular_author = {
        'name': author_popularity.index[0],
        'book_count': int(author_popularity.iloc[0])
    }

    # Visualization
    def create_enhanced_visualization():
        # Set up a cool, soft red color palette
        plt.figure(figsize=(14, 7), dpi=150)
        plt.style.use('seaborn-v0_8-white')
        
        # Define custom bin edges
        bins = [0, 4, 5, 6, 7, 8, 9, 10]
        
        # Create the histogram with soft red aesthetics
        hist_data = sns.histplot(
            ratings['Rating'], 
            kde=False, 
            bins=bins, 
            color='#FFB6C1',  # Light Pink Red 
            alpha=0.8,
            edgecolor='#FF6B6B'  # Soft Coral Edge
        )
        
        # Adjust y-axis to show more detail
        plt.ylim(bottom=0, top=max(hist_data.containers[0].datavalues) * 1.2)
        
        # Professional typography and styling
        plt.title('Rating Distribution', 
                fontweight='bold', 
                fontsize=18, 
                color='#333333',
                pad=20)
        
        plt.xlabel('Rating', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Add subtle grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
        
        # Remove top and right spines for a cleaner look
        sns.despine(top=True, right=True)
        
        plt.tight_layout()
        
        # Save to base64 for embedding
        img = BytesIO()
        plt.savefig(img, 
                    format='png', 
                    bbox_inches='tight', 
                    facecolor='white',
                    edgecolor='none')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        
        return img_base64

    chart_image_base64 = create_enhanced_visualization()

    # Construct response
    response = {
        "overview": {
            "total_books": total_books,
            "total_authors": total_authors,
            "total_users": total_users,
            "most_rated_book": {
                "name": most_rated_book['Book Title'],
                "author": most_rated_book['Book Author'],
                "rating_count": int(most_rated_book['count']),
                "avg_rating": round(most_rated_book['mean'], 2)
            },
            "top_rated_book": {
                "name": top_rated_book['Book Title'],
                "author": top_rated_book['Book Author'],
                "avg_rating": round(top_rated_book['mean'], 2)
            },
            "most_active_user": {
                "user_id": most_active_user['User Id'],
                "rating_count": int(most_active_user['rating_count']),
                "avg_user_rating": round(most_active_user['avg_rating'], 2)
            },
            "most_popular_author": most_popular_author
        },
        "visuals": {
            "chart_image": f"data:image/png;base64,{chart_image_base64}",
            "chart_description": "Rating Distribution Visualization"
        }
    }
    return render_template('overview.html', overview=response['overview'], visuals=response['visuals'])

@app.route('/top-authors')
def top_authors():
    required_books_columns = {'ISBN', 'Book Author'}
    required_ratings_columns = {'ISBN', 'Rating'}
    
    missing_books_columns = required_books_columns - set(books_wo_r_df.columns)
    missing_ratings_columns = required_ratings_columns - set(ratings.columns)
    
    if missing_books_columns or missing_ratings_columns:
        return f"Error: Missing columns - Books: {missing_books_columns}, Ratings: {missing_ratings_columns}", 500

    insights = analyze_books_dynamic(books_wo_r_df, ratings)
    
    # Book Counts and Average Ratings Data
    book_counts = books_wo_r_df['Book Author'].value_counts().reset_index()
    book_counts.columns = ['author', 'book_count']
    
    author_ratings = insights.reset_index()
    author_ratings.columns = ['author', 'avg_rating']
    
    # Merging book counts and ratings
    top_authors_detailed = book_counts.merge(
        author_ratings,
        on='author',
        how='left'
    )
    
    # Cleaning up data
    top_authors_detailed['avg_rating'] = top_authors_detailed['avg_rating'].round(2)
    top_authors_detailed = top_authors_detailed.sort_values('book_count', ascending=False).head(10)
    top_authors_list = top_authors_detailed.to_dict('records')
    
    # Identifying most prolific and highest rated
    most_prolific = top_authors_detailed.iloc[0]
    highest_rated = top_authors_detailed.nlargest(1, 'avg_rating').iloc[0]
    top_5_authors = [{'author': author['author'], 'book_count': author['book_count']} for author in top_authors_list[:5]]

    # Generating the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(top_authors_detailed['author'], top_authors_detailed['book_count'], color='skyblue')
    plt.title('Top 10 Authors by Number of Books', fontsize=15)
    plt.xlabel('Authors', fontsize=12)
    plt.ylabel('Number of Books', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Passing the data to the template
    return render_template(
        'top_authors.html',
        insights={
            'most_books_author': most_prolific['author'],
            'most_books_count': most_prolific['book_count'],
            'top_5_authors': top_5_authors,
            'highest_avg_rating_author': highest_rated['author'],
            'highest_avg_rating': highest_rated['avg_rating']
        },
        chart_img=f"data:image/png;base64,{chart_image}",
        top_authors=top_authors_list
    )

@app.route('/yearly-trends')
def yearly_trends():
    # Ensure we're working with valid publication years
    books_with_valid_years = books_wo_r_df[
        (books_wo_r_df['Year of Publication'] >= 1950) &
        (books_wo_r_df['Year of Publication'] <= 2024)
    ]
    
    # Create a complete DataFrame with all years from 1900 to 2023
    all_years = pd.DataFrame({'Year of Publication': range(1950, 2024)})
    
    # Count publications per year
    yearly_publications = books_with_valid_years['Year of Publication'].value_counts().sort_index()
    
    # Merge with all years, filling missing years with 0
    complete_yearly_publications = all_years.merge(
        yearly_publications.reset_index(), 
        on='Year of Publication', 
        how='left'
    ).fillna(0)
    
    # Rename columns
    complete_yearly_publications.columns = ['Year of Publication', 'Publications']
    
    # Sort by year
    complete_yearly_publications = complete_yearly_publications.sort_values('Year of Publication')
    
    # Prepare insights about yearly trends
    insights = {
        'peak_publication_year': complete_yearly_publications.loc[complete_yearly_publications['Publications'].idxmax(), 'Year of Publication'],
        'peak_publication_count': int(complete_yearly_publications['Publications'].max()),
        'lowest_publication_year': complete_yearly_publications[complete_yearly_publications['Publications'] > 0]['Year of Publication'].min(),
        'lowest_publication_count': int(complete_yearly_publications[complete_yearly_publications['Publications'] > 0]['Publications'].min()),
        'total_publications': int(complete_yearly_publications['Publications'].sum()),
        'total_years': len(complete_yearly_publications)
    }
    
    # Prepare data for Chart.js
    chart_years = list(map(str, complete_yearly_publications['Year of Publication']))
    chart_publications = list(complete_yearly_publications['Publications'])
    
    # Calculate additional trend insights
    trend_insights = analyze_publication_trends(books_with_valid_years)
    
    # Corrected variables passed to the template
    return render_template('yearly_trends.html',
        chart_years=chart_years,
        chart_publications=chart_publications,
        peak_publication_year=insights['peak_publication_year'],
        lowest_publication_year=insights['lowest_publication_year'],
        total_publications=insights['total_publications'])

def analyze_publication_trends(books_df):
    """
    Perform deeper analysis of publication trends
    """
    # Calculate year-over-year growth
    yearly_publications = books_df['Year of Publication'].value_counts().sort_index()
    yoy_growth = yearly_publications.pct_change() * 100
    
    # Find significant periods or trends
    trend_insights = {
        'average_annual_growth': yoy_growth.mean(),
        'most_dynamic_period': {
            'start_year': yoy_growth.idxmax(),
            'growth_rate': yoy_growth.max()
        },
        'periods_of_decline': list(yoy_growth[yoy_growth < 0].index)
    }
    
    return trend_insights

@app.route('/genre-analysis', methods=['GET', 'POST'])
def genre_analysis():
    book_details = None
    try:
        if request.method == 'POST':
            book_title = request.form['book_title']
            print(f"Received book title: {book_title}")  # Debug print
            # Scrape book details using the search title
            book_details = scrape_book_details(book_title, books_wo_r)
            print(f"Book details: {book_details}")  # Debug print
            
            # Add a flash message if no details found
            if book_details is None:
                flash(f"Could not find details for '{book_title}'. Please try a different book.", 'warning')
    except Exception as e:
        logger.error(f"Error in genre analysis route: {e}")
        flash("An unexpected error occurred. Please try again.", 'error')
    
    return render_template('genre_analysis.html', book_details=book_details)

@app.route('/ratings-analysis')
def ratings_analysis():
    # Get ratings analysis insights
    analysis_insights = get_ratings_analysis(books_w_r, ratings)  
    
    return render_template(
        'ratings_analysis.html', 
        top_rated_books=analysis_insights['top_rated_books'],
        top_active_users=analysis_insights['top_active_users'],
        ratings_distribution=analysis_insights['ratings_distribution']
    )

@app.route('/users')
def users():
    """
    Render users page and provide user insights.
    """
    # Print full path for debugging
    print(f"Full Users Path: {USERS_PATH}")
    
    try:
        # Generate insights
        user_insights = get_user_insights(users_df)
        
        # Debug print insights
        print("User Insights:")
        print(user_insights)
        
        return render_template('users.html', user_insights=user_insights)
    except Exception as e:
        print(f"Error in users route: {e}")
        return "Error loading user insights", 500

@app.route('/api/user-insights')
def user_insights_api():
    """
    API endpoint to fetch user insights as JSON.
    """
    try:
        user_insights = get_user_insights(users_df)
        return jsonify(user_insights)
    except Exception as e:
        print(f"Error in user insights API: {e}")
        return jsonify({
            'error': 'Unable to load user insights',
            'details': str(e)
        }), 500

@app.route('/ml-model')
def ml_model():

    results, visuals = utils.run_ml_models(USERS_PATH, RATINGS_PATH, BOOKS_PATH)
    return render_template('ml_model.html', results=results, visuals=visuals)

@app.route('/collaborative-filtering', methods=['GET'])
def collaborative_filtering_route():
    try:
        # Load datasets
        ratings = load_ratings(RATINGS_PATH)
        books = load_books_w_r(BOOKS_PATH, ratings)
        
        # Get query parameters
        search_query = request.args.get('search_query', '').strip()
        author_query = request.args.get('author_query', '').strip()
        rating_query = request.args.get('rating_query', '').strip()
        
        # Perform selection based on query type
        if search_query:
            recommendations = get_books_by_search(search_query, books)
        elif author_query:
            recommendations = get_books_by_search(author_query, books)
        elif rating_query:
            recommendations = get_books_by_rating(rating_query, books)
        else:
            # Default: top 10 books by rating
            recommendations = (
                books.nlargest(10, 'Rating')[['Book Title', 'Book Author', 'Rating', 'ISBN']]
                .to_dict(orient='records')
            )
        
        # Perform collaborative filtering
        cf_results = collaborative_filtering(ratings, books, search_query or author_query or rating_query)
        
        # Render template with results
        return render_template('collaborative_filtering.html',
            recommendations=recommendations or cf_results.get('recommendations', []),
            similar_books=cf_results.get('similar_books', []),
            search_query=search_query,
            author_query=author_query,
            rating_query=rating_query,
            user_based_rmse=cf_results.get('user_based_rmse', 0),
            item_based_rmse=cf_results.get('item_based_rmse', 0),
            error=cf_results.get('error')
        )
    except Exception as e:
        # Log the error and return an error page
        print(f"Error in collaborative filtering route: {str(e)}")
        return render_template('collaborative_filtering.html',
            recommendations=[],
            similar_books=[],
            error=f"An error occurred: {str(e)}"
        )

if __name__ == '__main__':
    # Initialize database
    with app.app_context():
        init_db()
    
    # Run the app
    app.run(debug=True)
