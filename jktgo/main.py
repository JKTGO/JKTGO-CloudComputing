from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import logging
import numpy as np

app = Flask(__name__)

class DataLoader:
    def __init__(self):
        # Load dataset
        self.tourism_data = pd.read_csv('wisata.csv')
        self.hotel_data = pd.read_csv('hotel.csv')

        # Initialize encoders and scalers
        self.tourism_category_encoder = LabelEncoder()
        self.hotel_region_encoder = LabelEncoder()
        self.price_scaler = MinMaxScaler()
        self.rating_scaler = MinMaxScaler()

        # Preprocess data
        self.preprocess_data()

    def preprocess_data(self):
        # Clean and encode tourism data
        self.tourism_data['Price'] = pd.to_numeric(self.tourism_data['Price'], errors='coerce')
        self.tourism_data['Rating'] = pd.to_numeric(self.tourism_data['Rating'], errors='coerce')
        self.tourism_data['Category_encoded'] = self.tourism_category_encoder.fit_transform(self.tourism_data['Category'])
        self.tourism_data['Price_scaled'] = self.price_scaler.fit_transform(self.tourism_data[['Price']])
        self.tourism_data['Rating_scaled'] = self.rating_scaler.fit_transform(self.tourism_data[['Rating']])

        # Clean and encode hotel data
        self.hotel_data['originalRate_perNight_totalFare'] = pd.to_numeric(self.hotel_data['originalRate_perNight_totalFare'], errors='coerce')
        self.hotel_data['starRating'] = pd.to_numeric(self.hotel_data['starRating'], errors='coerce')
        self.hotel_data['userRating'] = pd.to_numeric(self.hotel_data['userRating'], errors='coerce')
        self.hotel_data['region_encoded'] = self.hotel_region_encoder.fit_transform(self.hotel_data['region'])

class RecommenderModel:
    def __init__(self, input_dim, categorical_dim, num_categories):
        # Tambahkan regularization untuk mencegah overfitting
        self.model = self.build_model(input_dim, categorical_dim, num_categories)

    def build_model(self, input_dim, categorical_dim, num_categories):
        # Numerical features input
        numerical_input = Input(shape=(input_dim,))

        # Categorical features input
        categorical_input = Input(shape=(1,))

        # Embedding layer dengan regularisasi
        embedding = Embedding(num_categories, 8,
                              embeddings_regularizer=l2(0.001))(categorical_input)
        flatten = Flatten()(embedding)

        # Combine numerical and categorical features
        concat = Concatenate()([numerical_input, flatten])

        # Deep neural network layers dengan regularisasi
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(concat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        output = Dense(1, activation='sigmoid')(dropout2)

        model = Model(inputs=[numerical_input, categorical_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def train(self, X_num, X_cat, y, epochs=500, batch_size=32, validation_split=0.2):
        return self.model.fit(
            [X_num, X_cat],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = load_model(filepath)

class TourismRecommender:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None
        self.train_model()

    def prepare_training_data(self):
        tourism_data = self.data_loader.tourism_data
        X_numerical = tourism_data[['Price_scaled', 'Rating_scaled']].values
        X_categorical = tourism_data['Category_encoded'].values
        # Create synthetic target variable based on popularity
        y = (tourism_data['Rating'] > tourism_data['Rating'].mean()).astype(int)
        return train_test_split(X_numerical, X_categorical, y, test_size=0.2)

    def train_model(self):
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = self.prepare_training_data()
        num_categories = len(self.data_loader.tourism_category_encoder.classes_)

        self.model = RecommenderModel(input_dim=2, categorical_dim=1, num_categories=num_categories)
        self.model.train(X_num_train, X_cat_train, y_train)
        self.model.save_model('tourism_recommender.h5')

    def get_recommendations(self, category=None, max_price=None, min_rating=None):
        recommendations = self.data_loader.tourism_data.copy()

        # Implementasi filter yang lebih fleksibel
        if category:
            # Gunakan pencarian substring untuk kategori
            recommendations = recommendations[
                recommendations['Category'].str.contains(category, case=False, na=False)
            ]

        # Filter berdasarkan max_price jika tidak None
        if max_price is not None:
            recommendations = recommendations[recommendations['Price'] <= max_price]

        if min_rating is not None:
            recommendations = recommendations[recommendations['Rating'] >= min_rating]

        # Perhitungan skor yang lebih komplek
        X_num = recommendations[['Price_scaled', 'Rating_scaled']].values
        X_cat = recommendations['Category_encoded'].values
        predictions = self.model.predict([X_num, X_cat])

        recommendations['pred_score'] = predictions
        recommendations['weighted_score'] = (
            recommendations['pred_score'] * 0.5 +
            recommendations['Rating_scaled'] * 0.3 +
            (1 - recommendations['Price_scaled']) * 0.2
        )

        recommendations = recommendations.sort_values('weighted_score', ascending=False)

        return recommendations[['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Coordinate', 'Description', 'image']].head(10)

    def get_all(self):
        recommendations = self.data_loader.tourism_data.copy()
        return recommendations[['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Coordinate', 'Description', 'image']]

class HotelRecommender:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None
        self.train_model()

    def prepare_training_data(self):
        hotel_data = self.data_loader.hotel_data
        X_numerical = np.column_stack((
            hotel_data['starRating'],
            hotel_data['userRating'],
            hotel_data['originalRate_perNight_totalFare']
        ))
        X_categorical = hotel_data['region_encoded'].values
        # Create synthetic target variable based on user ratings
        y = (hotel_data['userRating'] > hotel_data['userRating'].mean()).astype(int)
        return train_test_split(X_numerical, X_categorical, y, test_size=0.2)

    def train_model(self):
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = self.prepare_training_data()
        num_regions = len(self.data_loader.hotel_region_encoder.classes_)

        self.model = RecommenderModel(input_dim=3, categorical_dim=1, num_categories=num_regions)
        self.model.train(X_num_train, X_cat_train, y_train)
        self.model.save_model('hotel_recommender.h5')

    def get_recommendations(self, star_rating=None, max_price=None, min_user_rating=None, region=None):
        recommendations = self.data_loader.hotel_data.copy()

        # Filter yang lebih fleksibel
        if star_rating:
            recommendations = recommendations[recommendations['starRating'] == star_rating]

        if max_price is not None:
            recommendations = recommendations[recommendations['originalRate_perNight_totalFare'] <= max_price]

        if min_user_rating is not None:
            recommendations = recommendations[recommendations['userRating'] >= min_user_rating]

        if region:
            recommendations = recommendations[recommendations['region'].str.contains(region, case=False, na=False)]

        # Perhitungan skor
        X_num = np.column_stack((
            recommendations['starRating'],
            recommendations['originalRate_perNight_totalFare'],
            recommendations['userRating']
        ))
        X_cat = recommendations['region_encoded'].values
        predictions = self.model.predict([X_num, X_cat])

        # Tambahkan skor dan urutkan
        recommendations['pred_score'] = predictions
        recommendations['weighted_score'] = (
            recommendations['pred_score'] * 0.4 +
            (1 - recommendations['originalRate_perNight_totalFare'] / recommendations['originalRate_perNight_totalFare'].max()) * 0.3 +
            recommendations['userRating'] * 0.3
        )

        recommendations = recommendations.sort_values('weighted_score', ascending=False)

        return recommendations[['name', 'region', 'starRating', 'userRating', 'originalRate_perNight_totalFare', 'hotelFeatures', 'image', 'Description']].head(10)
    
    def get_all(self):
        recommendations = self.data_loader.hotel_data.copy()
        return recommendations[['name', 'region', 'starRating', 'userRating', 'originalRate_perNight_totalFare', 'hotelFeatures', 'image', 'Description']]

class TextPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('indonesian'))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def extract_entities(self, text):
        entities = {
            'category': None,
            'price_range': None,
            'rating': None,
            'star_rating': None,
            'region': None
        }

        # Perluas daftar kategori dan wilayah
        categories = [
            'budaya', 'taman hiburan', 'bahari', 'cagar alam', 'pusat perbelanjaan',
            'sejarah', 'religi', 'kuliner', 'belanja', 'pantai', 'gunung', 'air terjun'
        ]
        regions = [
            'menteng', 'senayan', 'ancol', 'kemayoran', 'sudirman',
            'jakarta', 'bogor', 'depok', 'tangerang', 'bekasi'
        ]

        # Implementasi pencarian kategori dan wilayah yang lebih fleksibel
        text_lower = text.lower()
        matched_categories = [cat for cat in categories if cat in text_lower]
        matched_regions = [reg for reg in regions if reg in text_lower]

        # Prioritaskan kategori dan wilayah yang paling cocok
        entities['category'] = matched_categories[0] if matched_categories else None
        entities['region'] = matched_regions[0] if matched_regions else None

        # Perbaikan deteksi rentang harga dengan regex yang lebih komprehensif
        price_patterns = [
            (r'harga murah|budget|termurah', (0, 50000)),
            (r'harga sedang|menengah', (50000, 200000)),
            (r'harga mahal|mewah', (200000, float('inf')))
        ]

        for pattern, price_range in price_patterns:
            if re.search(pattern, text_lower):
                entities['price_range'] = price_range
                break

        # Perbaikan deteksi rating dengan regex yang lebih presisi
        rating_patterns = [
            r'rating minimal (\d+(?:\.\d+)?)',
            r'minimal rating (\d+(?:\.\d+)?)',
            r'rating di atas (\d+(?:\.\d+)?)'
        ]

        for pattern in rating_patterns:
            rating_match = re.search(pattern, text_lower)
            if rating_match:
                entities['rating'] = float(rating_match.group(1))
                break

        # Deteksi rating bintang untuk hotel
        star_patterns = [
            r'(\d+) bintang',
            r'hotel bintang (\d+)',
            r'bintang (\d+)'
        ]

        for pattern in star_patterns:
            star_match = re.search(pattern, text_lower)
            if star_match:
                entities['star_rating'] = int(star_match.group(1))
                break

        return entities

class TourismChatbot:
    def __init__(self):
        logging.basicConfig(
            filename='chatbot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def handle_user_input(self, user_input):
        try:
            entities = text_preprocessor.extract_entities(user_input)

            if 'hotel' in user_input.lower():
                recommendations = hotel_recommender.get_recommendations(
                    star_rating=entities['star_rating'],
                    region=entities['region']
                )
            else:
                recommendations = tourism_recommender.get_recommendations(
                    category=entities['category']
                )

            return self.format_recommendations(recommendations, 'hotel' in user_input.lower())
        except Exception as e:
            logging.error(str(e))
            return {"message": "Maaf, terjadi kesalahan."}

    def format_recommendations(self, recommendations, is_hotel):
        if is_hotel:
            return recommendations.to_dict('records')
        return recommendations.to_dict('records')

# Load data dan model
data_loader = DataLoader()
tourism_recommender = TourismRecommender(data_loader)
hotel_recommender = HotelRecommender(data_loader)
hotel_recommender.model = load_model('hotel_recommender.h5')
tourism_recommender.model = load_model('tourism_recommender.h5')
text_preprocessor = TextPreprocessor()
chatbot = TourismChatbot()

# Error Handler

@app.errorhandler(404)
def not_found_error(e):
    return jsonify({
        "error": "The requested URL was not found on the server.",
        "status_code": 404
    }), 404

@app.errorhandler(400)
def bad_request_error(e):
    return jsonify({
        "error": "Bad Request. The server could not understand the request due to invalid syntax.",
        "status_code": 400
    }), 400

@app.errorhandler(405)
def method_not_allowed_error(e):
    return jsonify({
        "error": f"The method {request.method} is not allowed for the requested URL.",
        "status_code": 405
    }), 405

@app.errorhandler(415)
def unsupported_media_type_error(e):
    return jsonify({
        "error": "Unsupported Media Type. Make sure the Content-Type is 'application/json'.",
        "status_code": 415
    }), 415

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({
        "error": "Internal Server Error. An unexpected condition was encountered.",
        "status_code": 500
    }), 500

# @app.errorhandler(ValueError)
# def handle_value_error(error):
#     return jsonify({"message": "Terjadi kesalahan pada input yang diberikan.", "error": str(error)}), 400

# GET

@app.route('/tourism', methods=['GET'])
def get_all_tourism():
    recommendations = tourism_recommender.get_all()
    return jsonify(recommendations.to_dict('records')), 200

@app.route('/hotel', methods=['GET'])
def get_all_hotel():
    recommendations = hotel_recommender.get_all()
    return jsonify(recommendations.to_dict('records')), 200

# POST

@app.route('/recommend_tourism', methods=['POST'])
def recommend_tourism():
    data = request.get_json()

    if not data or not any(data.values()):
        return jsonify({
            "error": "Request body must contain at least one valid field."
        }), 400

    category = data.get('category')
    max_price = data.get('max_price')
    min_rating = data.get('min_rating')

    recommendations = tourism_recommender.get_recommendations(
        category=category,
        max_price=max_price,
        min_rating=min_rating
    )

    return jsonify(recommendations.to_dict('records')), 200

@app.route('/recommend_hotel', methods=['POST'])
def recommend_hotel():
    data = request.get_json()

    if not data or not any(data.values()):
        return jsonify({
            "error": "Request body must contain at least one valid field."
        }), 400

    star_rating = data.get('star_rating')
    max_price = data.get('max_price')
    min_user_rating = data.get('min_user_rating')
    region = data.get('region')

    recommendations = hotel_recommender.get_recommendations(
        star_rating=star_rating,
        max_price=max_price,
        min_user_rating=min_user_rating,
        region=region
    )

    return jsonify(recommendations.to_dict('records')), 200

@app.route('/preprocess_text', methods=['POST'])
def preprocess_text():
    data = request.get_json()
    user_input = data.get('text')
    entities = chatbot.handle_user_input(user_input)
    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)