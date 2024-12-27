import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import logging
from flask import Flask
from utils import load_books_w_r, load_ratings, get_secret_key, load_ratings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = get_secret_key()

if __name__ == "__main__":
    file_path = '/Users/archits/Downloads/book-recommendation-system/cleaned_datasets'
    ratings_path = f'{file_path}/cleaned_ratings.csv'
    books_path = f'{file_path}/cleaned_books.csv'

    # Load datasets
    ratings = load_ratings(ratings_path)
    books_w_r = load_books_w_r(books_path, ratings)
    print(books_w_r.columns)
