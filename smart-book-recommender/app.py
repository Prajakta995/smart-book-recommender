import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
books = pd.read_csv("realistic_books_full.csv")

# Combine fields for AI similarity matching
books["combined"] = (
    books["description"].fillna("") + " " +
    books["genre"].fillna("") + " " +
    books["author"].fillna("")
)

# Vectorize book features
vectorizer = TfidfVectorizer(stop_words='english')
book_vectors = vectorizer.fit_transform(books["combined"])

# ---------- UI Setup ----------
st.set_page_config(page_title="📚 Smart Book Recommender", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>📚 Smart Book Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Tell us your story mood and get perfect book matches!</p>", unsafe_allow_html=True)

# ---------- User Input ----------
user_input = st.text_area("✏️ Describe the kind of story you want:", height=100, placeholder="e.g., a space adventure with friendship and discovery")

col1, col2 = st.columns(2)
with col1:
    genres = sorted(books["genre"].unique())
    selected_genre = st.selectbox("🎨 Optional: Filter by Genre", ["All"] + genres)

with col2:
    year_range = st.slider("📅 Optional: Year Range", 1850, 2023, (2000, 2023))

# ---------- Recommendation Logic ----------
if st.button("🔍 Recommend Books"):
    if user_input.strip():
        input_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(input_vec, book_vectors).flatten()

        # Genre & Year Filter
        filtered_books = books.copy()
        if selected_genre != "All":
            filtered_books = filtered_books[filtered_books["genre"] == selected_genre]
        filtered_books = filtered_books[
            (filtered_books["year"] >= year_range[0]) & (filtered_books["year"] <= year_range[1])
        ]

        # Recalculate similarity with filtered books
        filtered_indices = filtered_books.index
        filtered_similarity = similarity[filtered_indices]
        top_indices = filtered_similarity.argsort()[-5:][::-1]

        st.subheader("🎯 Top Book Matches")
        for i in top_indices:
            idx = filtered_indices[i]
            book = books.iloc[idx]
            with st.container():
                st.markdown(f"""
                <div style="border:2px solid #ddd; border-radius:12px; padding:16px; margin-bottom:16px; background-color:#f9f9f9;">
                    <h3 style="color:#333;">{book['title']}</h3>
                    <p><strong>Author:</strong> {book['author']} &nbsp; | &nbsp;
                    <strong>Genre:</strong> {book['genre']} &nbsp; | &nbsp;
                    <strong>Year:</strong> {book['year']}</p>
                    <p style="font-size: 15px; color:#555;">{book['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to describe the kind of book you want.")

