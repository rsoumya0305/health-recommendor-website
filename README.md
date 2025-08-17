# health-recommendor-website
This project is a Machine Learning + Full Stack Web Application built with Flask that predicts diseases based on user-provided symptoms. The system not only identifies the most likely disease but also provides additional health-related insights, including the disease description, recommended precautions, medications, workouts, and diet plans, making it a comprehensive health recommender system.

The backend is powered by a Machine Learning model trained on a curated dataset (health_data_final.csv) containing multiple diseases and their associated details. The model is serialized into a model.pkl file, which is then used by the Flask app to make real-time predictions. On the frontend, the application provides a clean and user-friendly web interface where users can input their symptoms and instantly get tailored health recommendations.

The project structure includes the Flask app (app.py), the model training script (train_model.py), datasets stored in the data folder, templates for the user interface inside the templates directory, and static assets such as styles and images in the static directory. All dependencies are listed in requirements.txt.
