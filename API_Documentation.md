## CMPT 2500 Project – Vehicle Sales Clustering Prediction
## Overview

This API serves a machine-learning model that clusters vehicle-sales records to identify best-selling regions across Canada.
It exposes endpoints to check API health, get usage information, and receive cluster assignments from two different model versions (v1 and v2).

This document provides instructions for setup and a high-level overview of the endpoints. For a detailed, interactive API reference, run the server and navigate to /apidocs/.
 1. Setup Environment
Clone the repository and install the required Python packages.
```sh
git clone https://github.com/[YOUR_USERNAME]/cmpt2500f25-project-cluster-driver-Lab2.git  
cd cmpt2500f25-project-cluster-driver-Lab2  
python -m venv .venv  
source .venv/bin/activate    # Windows: .venv\Scripts\activate  
pip install -r requirements.txt  

2. Get Data & Artifacts
This project uses DVC to manage large data files and pipelines. You must
set up your DagsHub credentials and pull the data.
(Follow the credential setup in `assignments/Lab 03 - REST API
Development.md` if this is a new environment).
```sh
dvc pull data/processed
```
This will download 'data/processed/final_features.parquet', 'models/model_v1.pkl' and `models/model_v2.pkl`.
### 3. Run the API Server
Ensure your `models/model_v1.pkl` and `models/model_v2.pkl` files are in
place. Then, run the app:
```sh
python src/app.py
```
The server will start on `http://127.0.0.1:5000`.
---
## Endpoints
`GET /health`
- **Purpose**: A simple health check.
- **Success Response (200 OK)**:
```json
Lab 03 - REST API Development.md 2025-10-31
28 / 31
{
  "message": "Welcome to the Vehicle Sales Clustering API!",  
  "api_documentation": "Use /apidocs for interactive Swagger UI.",  
  "endpoints": {  
    "health_check": "/health",  
    "predict_v1": "/v1/predict",  
    "predict_v2": "/v2/predict"  
  },  
  "required_input_format": {  
    "FSA_Code": "T2A",  
    "make": "Toyota",  
    "price": 18500,  
    "Region": "Prairies",  
    "Most_sold_brand": "Toyota",  
    "Average_mileage": 120000,  
    "Average_price": 18000,  
    "FSA_Latitude": 51.05,  
    "FSA_Longitude": -114.07,  
    "Region_vehicle_sold": 5000,  
    "Region_dealerships": 40,  
    "Most_sold_month": "June"  
  }  
}

{
  "FSA_Code": "T2A",  
  "make": "Toyota",  
  "price": 18500,  
  "Region": "Prairies",  
  "Most_sold_brand": "Toyota",  
  "Average_mileage": 120000,  
  "Average_price": 18000,  
  "FSA_Latitude": 51.05,  
  "FSA_Longitude": -114.07,  
  "Region_vehicle_sold": 5000,  
  "Region_dealerships": 40,  
  "Most_sold_month": "June"  
}

```
- **Success Response (200 OK)**:
```json
{
  "cluster": 3,  
  "model_version": "v1"  
}

```
- **Error Response (400 Bad Request)**:
```json
{"error": "Missing required features: FSA_Code, Region, ..."}  

```
`GET /apidocs/`
- **Purpose**: Provides a full, interactive "Swagger UI" for the API. You
can see all endpoints, data models, and test them live from your browser.
