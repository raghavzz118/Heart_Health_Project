# Heart Disease Risk Prediction System

An end-to-end machine learning classification system designed to assess cardiovascular disease risk using clinical patient data. The system optimizes model evaluation towards high recall to minimize critical false negatives in medical diagnostics, served via a Flask web API and deployed on AWS Elastic Beanstalk.

---

## Key Highlights

* **Model Performance:** Achieved a **74% recall rate** on **4,000+ patient records**, specifically tuning the classifier to reduce false negatives.
* **Production Deployment:** Containerized and deployed as a production-ready **Flask API** on **AWS Elastic Beanstalk** for scalable, real-time inference.
* **Modular Codebase:** Organized into modular pipelines (`src/`), custom configurations (`.ebextensions`), and serializable model artifacts (`artifacts/`).

---

## Architecture & Workflow

1. **Data Ingestion & Preprocessing:** Handles missing values, performs feature scaling, and structures tabular clinical data.
2. **Model Training & Evaluation:** Trains classification pipelines using Scikit-learn with focus on maximizing sensitivity (recall) for clinical safety.
3. **Web Application:** Built an interactive web frontend using HTML templates and Flask routes (`application.py`).
4. **Cloud Deployment:** Configured automated deployment pipelines onto AWS Elastic Beanstalk.

---