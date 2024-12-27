import pandas as pd
import os
import re
import textwrap
import aiohttp
import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm as asyncio_tqdm

# Comprehensive genre list
GENRES = {
    'Fiction': [
        'Literary Fiction', 'Contemporary Fiction', 'Historical Fiction',
        'Science Fiction', 'Fantasy', 'Mystery', 'Thriller', 'Romance',
        'Horror', 'Adventure', 'Crime Fiction', 'Young Adult', 
        'Children\'s Literature', 'Drama', 'Poetry'
    ],
    'Non-Fiction': [
        'Biography', 'Autobiography', 'Memoir', 'History', 'Science',
        'Philosophy', 'Psychology', 'Self-Help', 'Business', 'Economics',
        'Politics', 'Social Sciences', 'Technology', 'Education',
        'Travel', 'Cooking', 'Health', 'Religion', 'Art', 'Music'
    ]
}

# Flatten genres for easy lookup
ALL_GENRES = [genre for category in GENRES.values() for genre in category]

class GenreCache:
    def __init__(self, filename: str = 'genre_cache.csv'):
        self.filename = filename
        self.cache: Dict[str, dict] = {}
        self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            try:
                df = pd.read_csv(self.filename)
                self.cache = {
                    row['book_title']: {
                        'genre': row['genre'],
                        'confidence': row['confidence'],
                        'timestamp': datetime.fromisoformat(row['timestamp'])
                    }
                    for _, row in df.iterrows()
                }
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
    
    def save(self):
        try:
            df = pd.DataFrame([
                {
                    'book_title': title,
                    'genre': data['genre'],
                    'confidence': data['confidence'],
                    'timestamp': data['timestamp'].isoformat()
                }
                for title, data in self.cache.items()
            ])
            df.to_csv(self.filename, index=False)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def get(self, title: str) -> Optional[dict]:
        if title in self.cache:
            data = self.cache[title]
            if (datetime.now() - data['timestamp'] < timedelta(days=30) and 
                data['confidence'] >= 0.7):
                return data
        return None
    
    def set(self, title: str, genre: str, confidence: float):
        self.cache[title] = {
            'genre': genre,
            'confidence': confidence,
            'timestamp': datetime.now()
        }

class GenreClassifier:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = GenreCache()
        self.genre_patterns = self._compile_genre_patterns()
    
    def _compile_genre_patterns(self) -> Dict[str, List[re.Pattern]]:
        patterns = {}
        for genre in ALL_GENRES:
            variations = [
                genre.lower(),
                genre.replace(' ', '-').lower(),
                genre.replace(' ', '').lower()
            ]
            patterns[genre] = [re.compile(rf"\b{var}\b") for var in variations]
        return patterns
    
    def _extract_genres_from_text(self, text: str) -> List[tuple]:
        text = text.lower()
        matches = []
        
        for genre, patterns in self.genre_patterns.items():
            count = sum(1 for pattern in patterns if pattern.search(text))
            if count > 0:
                matches.append((genre, count))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    async def fetch_wikipedia_genres(self, session: ClientSession, title: str) -> tuple:
        clean_title = re.sub(r'[^\w\s]', '', title)
        search_title = clean_title.replace(' ', '_')
        
        urls = [
            f"https://en.wikipedia.org/wiki/{search_title}",
            f"https://en.wikipedia.org/wiki/{search_title}_(book)",
            f"https://en.wikipedia.org/wiki/Special:Search?search={search_title}+book"
        ]
        
        for url in urls:
            try:
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status != 200:
                        continue
                        
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    genre_matches = []
                    infobox = soup.find('table', class_='infobox')
                    if infobox:
                        genre_row = infobox.find('th', string=lambda x: x and 'Genre' in x)
                        if genre_row:
                            genre_cell = genre_row.find_next('td')
                            if genre_cell:
                                genre_matches.extend(self._extract_genres_from_text(genre_cell.get_text()))
                    
                    if not genre_matches:
                        paragraphs = soup.select('div.mw-parser-output > p')[:3]
                        for p in paragraphs:
                            genre_matches.extend(self._extract_genres_from_text(p.get_text()))
                    
                    if genre_matches:
                        top_genre = genre_matches[0][0]
                        confidence = min(1.0, genre_matches[0][1] / sum(count for _, count in genre_matches))
                        return top_genre, confidence
                        
            except Exception as e:
                continue
                
        return None, 0.0
    
    def _get_default_genre(self, title: str, summary: str = "") -> tuple:
        combined_text = f"{title} {summary}"
        matches = self._extract_genres_from_text(combined_text)
        
        if matches:
            return matches[0][0], 0.5
        return "Fiction", 0.3
    
    async def get_genre(self, session: ClientSession, title: str, summary: str = "") -> str:
        cached = self.cache.get(title)
        if cached:
            return cached['genre']
        
        genre, confidence = await self.fetch_wikipedia_genres(session, title)
        
        if not genre:
            genre, confidence = self._get_default_genre(title, summary)
        
        self.cache.set(title, genre, confidence)
        return genre

async def process_books_async(books_df: pd.DataFrame, chunk_size: int = 1000):
    classifier = GenreClassifier()
    
    # Ensure the 'Summary' column exists
    if 'Summary' not in books_df.columns:
        books_df['Summary'] = ""
    
    unique_books = books_df[['Book Title', 'Summary']].drop_duplicates()
    
    genres_dict = {}
    async with ClientSession() as session:
        for i in range(0, len(unique_books), chunk_size):
            chunk = unique_books.iloc[i:i+chunk_size]
            tasks = []
            for _, row in chunk.iterrows():
                title = row['Book Title']
                summary = str(row.get('Summary', ''))
                tasks.append(classifier.get_genre(session, title, summary))
            
            print(f"Processing chunk {i // chunk_size + 1}...")
            genres = await asyncio_tqdm.gather(*tasks)
            
            for title, genre in zip(chunk['Book Title'], genres):
                genres_dict[title] = genre
    
    classifier.cache.save()
    return genres_dict

def add_genres_to_books(file_path: str):
    try:
        books_df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip", low_memory=False)
        print(f"Loaded {len(books_df)} total book entries")
        print(f"Columns in dataset: {list(books_df.columns)}")

        # Ensure 'Summary' column exists
        if 'Summary' not in books_df.columns:
            books_df['Summary'] = ""

        # Process unique books and get genres
        genres_dict = asyncio.run(process_books_async(books_df))

        # Map genres back to the full dataset
        books_df['Genre'] = books_df['Book Title'].map(genres_dict)

        # Verify the mapping
        print(f"Added genres for {len(books_df[books_df['Genre'].notna()])} books")
        print(f"Missing genres for {len(books_df[books_df['Genre'].isna()])} books")

        output_file = 'books_with_improved_genres.csv'
        books_df.to_csv(output_file, index=False)
        print(f"Genres added and saved to {output_file}")

    except Exception as e:
        print(f"Error processing books: {e}")
        raise

if __name__ == "__main__":
    file_path = '/Users/archits/Downloads/book-recommendation-system/cleaned_datasets/cleaned_books.csv'  
    add_genres_to_books(file_path)
