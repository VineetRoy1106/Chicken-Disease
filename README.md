# MLOps Project

**Food Delivery services like Zomato and Swiggy need to show the accurate time it will take to deliver your order to keep transparency with their customers. These companies use Machine Learning algorithms to predict the food delivery time based on how much time the delivery partners took for the same distance in the past.**

**To predict the food delivery time in real-time, we need to calculate the distance between the food preparation point and the point of food consumption. After finding the distance between the restaurant and the delivery locations, we need to find relationships between the time taken by delivery partners to deliver the food in the past for the same distance.**

## 📦 Project Overview
The project is a Flask-based web application that allows users to:
- **Perform Single Data Point Prediction:** Using a web form.
- **Perform Batch Prediction:** Uploading CSV files for batch inference.
- **Train the Model:** Re-train the model using the web interface.

The entire pipeline is managed using MLOps best practices, including monitoring, automation, and containerization.

---

## 📂 Key Components

### 📡 Web Application
- **Framework:** Flask
- **Frontend:** HTML templates (index, form, batch, and train pages)

### 📊 Data Pipeline
- **Data Ingestion, Transformation, and Training**
- **Prediction Pipeline:** Custom prediction pipeline for both single and batch prediction.

### 🛠️ Tools & Techniques Used
- **Flask:** Web framework for deploying the model as a RESTful API.
- **Airflow:** Used for orchestrating and monitoring tasks in the pipeline: 
  - Data Ingestion
  - Data Transformation
  - Model Training
- **Docker:** For containerization and reproducibility.
- **Logging:** Implemented with a dedicated logger for better traceability.

---

### 📏 Metrics
The project uses the Haversine distance formula to calculate the distance between two geographical points:
```python
def distance_numpy(self, df, lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = 0.5 - np.cos((df[lat2] - df[lat1]) * p) / 2 + 
        np.cos(df[lat1] * p) * np.cos(df[lat2] * p) * (1 - np.cos((df[lon2] - df[lon1]) * p)) / 2
    df['distance'] = 12734 * np.arccos(np.sort(a))
```

---

## 📈 Features
- **Single Data Point Prediction:** Users can input data through a web form.
- **Batch Prediction:** Upload CSV files for batch predictions.
- **Model Training:** Retrain the model using the web interface.
- **Automated Pipelines:** Managed using Airflow and Docker.
- **Logging:** Implemented for effective debugging and monitoring.

---

## 📦 Project Structure
```
├── src
│   ├── components
│   ├── config
│   ├── logger
│   ├── pipeline
│   └── prediction
├── templates
│   ├── index.html
│   ├── form.html
│   ├── batch.html
│   └── train.html
├── batch_prediction
├── Dockerfile
├── requirements.txt
└── app.py
```

---

## 📷 Screenshots
### Delivery App Home
![deliverytime](https://github.com/user-attachments/assets/174c9a02-9e11-4098-b596-3483478198fb)

### Single Prediction Page
![deliverysingle](https://github.com/user-attachments/assets/206b26b5-8137-4652-b22a-ef68bb0059dc)

### Batch Prediction Page
![Batch Prediction](https://github.com/user-attachments/assets/221e21eb-3052-40c5-af38-cf9e8300b1b2)

---

## 🚀 Getting Started

### Prerequisites:
- Python 3.8+
- Docker
- Apache Airflow

### Setup:
1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-name>
```
2. **Create a virtual environment and install dependencies:**
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
3. **Run the Flask App:**
```bash
python app.py
```
4. **Access the Web App:** Open `http://localhost:8888`

---

## 📦 Docker Integration
- Build the Docker image:
```bash
docker build -t mlops-project .
```
- Run the container:
```bash
docker run -p 8888:8888 mlops-project
```

---

## 📊 Airflow Integration
- Start the Airflow service:
```bash
airflow standalone
```
- Access Airflow at: `http://localhost:8080`

---

## 📧 Contact
- **Email:** vineetroy289@gmail.com
- **LinkedIn:** [Vineet Roy](https://linkedin.com/in/vineet-roy-0370a61a9)

---

### 📜 License
This project is licensed under the MIT License.
