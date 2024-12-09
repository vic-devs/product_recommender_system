from flask import Flask, jsonify, request
import pandas as pd
import json
import pickle
from keras.models import load_model
from typing import List, Dict, Union
import logging
from keras.saving import register_keras_serializable
from tensorflow.keras.losses import MeanSquaredError

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for models and data
svd_model = None
knn_model = None
mlp_model = None
hybrid_weights = None
preferred_categories = None
products_df = None


@register_keras_serializable()
def mse(y_true, y_pred):
    mse_fn = MeanSquaredError()
    return mse_fn(y_true, y_pred)


# Load models and data
def load_models_and_data():
    global svd_model, knn_model, mlp_model, hybrid_weights, preferred_categories, products_df

    try:
        # Load SVD model
        with open('svd_model.pkl', 'rb') as svd_file:
            svd_model = pickle.load(svd_file)
            logger.info("SVD model loaded successfully")

        # Load KNN model
        with open('knn_model.pkl', 'rb') as knn_file:
            knn_model = pickle.load(knn_file)
            logger.info("KNN model loaded successfully")

        # Load MLP model with custom objects
        mlp_model = load_model('mlp_model.h5', custom_objects={'mse': mse})
        logger.info("MLP model loaded successfully")

        # Load hybrid weights and preferred categories
        with open('hybrid_weights.json', 'r') as weights_file:
            hybrid_weights = json.load(weights_file)
            logger.info("Hybrid weights loaded successfully")

        with open('preferred_categories.json', 'r') as categories_file:
            preferred_categories = json.load(categories_file)
            logger.info("Preferred categories loaded successfully")

        # Load products data
        products_df = pd.read_csv('products.csv')
        logger.info("Products data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models or data: {e}")
        raise e


# Hybrid recommendation logic
def get_hybrid_recommendations(user_id: int, top_n: int = 10, boost_factor: float = 1.2) -> Union[List[Dict], Dict]:
    try:
        if any(obj is None for obj in [svd_model, knn_model, mlp_model, hybrid_weights, preferred_categories, products_df]):
            raise ValueError("Models or data not loaded properly")

        recommendations = []
        user_categories = preferred_categories.get(str(user_id), [])
        if not isinstance(user_categories, list):
            user_categories = [user_categories]

        # Product-category mapping
        product_category_mapping = products_df.set_index('product_id')['category_id'].to_dict()

        # Generate recommendations
        for product_id in products_df['product_id'].unique():
            # SVD prediction
            svd_score = svd_model.predict(user_id, product_id).est

            # MLP prediction
            sample = pd.DataFrame({'user_id': [user_id], 'product_id': [product_id], 'interaction_value_normalized': [0]})
            mlp_input = sample[['user_id', 'product_id', 'interaction_value_normalized']].to_numpy()
            mlp_score = mlp_model.predict(mlp_input)[0][0]

            # KNN prediction (dummy value here, as sklearn KNN doesn't provide scoring directly)
            knn_score = 0.5  # Replace with KNN model prediction logic if needed

            # Combine scores
            hybrid_score = (
                hybrid_weights[0] * svd_score +
                hybrid_weights[1] * mlp_score +
                hybrid_weights[2] * knn_score
            )

            # Apply category boost
            product_category = product_category_mapping.get(product_id, None)
            if product_category in user_categories:
                hybrid_score *= boost_factor

            recommendations.append((product_id, product_category, hybrid_score))

        # Sort and format recommendations
        recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)[:top_n]
        result = [
            {
                'product_id': int(rec[0]),  # Convert to Python int
                'category_id': int(rec[1]) if rec[1] is not None else None,  # Convert to int if not None
                'score': float(rec[2])  # Convert to Python float
            }
            for rec in recommendations
        ]
        return result
    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {e}")
        return {"error": str(e)}


# Flask endpoint for hybrid recommendations
@app.route('/recommend/hybrid', methods=['GET'])
def recommend_hybrid():
    try:
        user_id = int(request.args.get('user_id'))
        top_n = int(request.args.get('top_n', 10))
        recommendations = get_hybrid_recommendations(user_id, top_n)

        if isinstance(recommendations, dict) and 'error' in recommendations:
            return jsonify(recommendations), 500

        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error in hybrid recommendation endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': all([svd_model, knn_model, mlp_model]),
        'data_loaded': products_df is not None
    })


if __name__ == '__main__':
    load_models_and_data()
    app.run(debug=True, port=7000)
