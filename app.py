import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding
from newspaper import Article
from flask import Flask, request, render_template

# Create Flask app
app = Flask(__name__)

# Define CNN Model
def create_cnn_model(vocab_size, embedding_dim, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))  # Adjust based on output
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Extract Article
def extract_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"An error occurred while extracting the article: {e}")
        return None

# Summarization Function using CNN (Placeholder)
def summarize_with_cnn(article_text, model, tokenizer, max_length):
    # Split the article into sentences
    sentences = article_text.split('. ')
    
    # If the article is very short (e.g., fewer than 3 sentences), return a shorter summary
    if len(sentences) <= 3:
        # Summarize the short article into a single sentence or key phrases
        summary = ' '.join(sentences[:1])  # Example: just the first sentence as the summary
        return summary
    
    # For longer articles, summarize by taking the first few sentences (adjust as needed)
    summary = ' '.join(sentences[:3])
    return summary

# Count words in text
def count_words(text):
    return len(text.split())

# Main Summarization Function
def summarize_news_article(article_text, model, tokenizer, max_length):
    if article_text:
        print("Summarizing the article now...\n")

        # Count words in the original article
        original_word_count = count_words(article_text)
        
        # Apply summarization for short articles as well
        summary = summarize_with_cnn(article_text, model, tokenizer, max_length)

        # Count words in the summary
        summarized_word_count = count_words(summary)

        return {
            "original_word_count": original_word_count,
            "summary": summary,
            "summarized_word_count": summarized_word_count
        }
    else:
        return None

# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle the input (either URL or pasted content) and provide summary
@app.route('/summarize', methods=['POST'])
def summarize():
    input_choice = request.form['input-choice']
    url = request.form.get('url')
    article_text = request.form.get('article-text')

    # Handle URL input
    if input_choice == 'url' and url:
        article_text = extract_article(url)
        if not article_text:
            return render_template('index.html', error="Failed to extract the article from the URL.")
    
    # Handle pasted article text
    elif input_choice == 'text' and article_text:
        pass  # article_text is already set from the form
    
    # Invalid input
    else:
        return render_template('index.html', error="Please provide a valid input.")

    # Define CNN model parameters
    vocab_size = 10000  # Example vocab size, adjust as needed
    embedding_dim = 100  # Embedding dimension
    max_length = 100  # Maximum length of input sequences

    # Create model (no training in this case)
    model = create_cnn_model(vocab_size, embedding_dim, max_length)

    # Summarize the article
    summary_result = summarize_news_article(article_text, model, None, max_length)

    if summary_result:
        return render_template('summary.html', summary=summary_result['summary'], 
                               original_word_count=summary_result['original_word_count'],
                               summarized_word_count=summary_result['summarized_word_count'])
    else:
        return render_template('index.html', error="Failed to summarize the article.")

# Run the Flask app 
if __name__ == "__main__":
    app.run(debug=True)
    
