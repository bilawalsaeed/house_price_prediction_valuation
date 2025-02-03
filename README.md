# House Price Prediction and Valuation (California Housing Dataset)

## 📌 Project Overview
This project predicts house prices using the **California Housing dataset** from **scikit-learn**. It applies **multivariable regression models**, including **Linear Regression, Ridge Regression, and Lasso Regression**, with advanced techniques such as:
- **Polynomial Feature Expansion**
- **Feature Scaling**
- **Hyperparameter Tuning with Grid Search**
- **Model Evaluation & Residual Analysis**
- **House Valuation for New Properties**

## 📊 Dataset Information
The dataset contains information on housing prices in different block groups in California. The target variable is **median house value** (in units of $100,000). The dataset includes the following features:
- `MedInc`: Median income in block group ($10,000s)
- `HouseAge`: Median house age (years)
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average household occupancy
- `Latitude`: Geographic latitude of block group
- `Longitude`: Geographic longitude of block group

## 🛠️ Tech Stack & Dependencies
- **Python 3.x**
- **NumPy** (for numerical operations)
- **Pandas** (for data handling)
- **Scikit-learn** (for machine learning models)
- **Matplotlib & Seaborn** (for visualization)



### 🚀 Running the Project
To execute the model training and evaluation:
```bash
python main.py
```

## 📈 Model Training & Evaluation
### 1️⃣ Models Implemented:
- **Linear Regression**: Baseline model
- **Ridge Regression**: L2 Regularization (Hyperparameter tuning: `alpha`)
- **Lasso Regression**: L1 Regularization (Hyperparameter tuning: `alpha` and polynomial degree)

### 2️⃣ Hyperparameter Tuning:
- Ridge and Lasso models are optimized using **Grid Search with Cross-Validation (CV=5)**.
- Best parameters are selected based on the **Mean Squared Error (MSE)**.

### 3️⃣ Performance Metrics:
- **Mean Squared Error (MSE)**
- **R² Score**
- **Residual Analysis & Distribution**
- **Actual vs Predicted Value Plot**

### 4️⃣ Figures:
- Correlation Matrix

![Figure_1](https://github.com/user-attachments/assets/63e923e8-3ec7-4a99-8e55-a02b15fc6359)

- Pair Plot (scatter matrix)

![Figure_2](https://github.com/user-attachments/assets/1814a804-e6b2-4917-aed2-47f2b3b4f7ab)

- Actual vs Predicted (Linear)

![Figure_3](https://github.com/user-attachments/assets/34fccf19-ce9b-46aa-a57e-29bda7e941f2)

- Residuals Distribution (Linear)

![Figure_4](https://github.com/user-attachments/assets/069a2f92-e2cf-4ac7-b4a0-202b3d3ce8d1)

- Actual vs Predicted (Ridge)

![Figure_5](https://github.com/user-attachments/assets/8e0e69d6-436d-4daf-bedd-92286198f061)

- Residuals Distribution (Ridge)

![Figure_6](https://github.com/user-attachments/assets/33139843-bd3e-4544-8840-29376cdefac9)

- Actual vs Predicted (Lasso)

![Figure_7](https://github.com/user-attachments/assets/a1c036b4-f197-47bf-9e1a-fbe68d71da9a)

- Residuals Distribution (Lasso)

![Figure_8](https://github.com/user-attachments/assets/41646fdb-17f5-4afc-973f-f1c9a0087f3a)


## 🏡 House Valuation Example
To predict the price of a new house, the model uses example inputs:
```python
new_house = {
    'MedInc': 8.0,         # Median income ($80,000)
    'HouseAge': 20,        # House age (years)
    'AveRooms': 6.0,       # Avg rooms per household
    'AveBedrms': 1.0,      # Avg bedrooms per household
    'Population': 1000,    # Block group population
    'AveOccup': 3.0,       # Avg occupants per household
    'Latitude': 34.0,      # Latitude
    'Longitude': -118.0    # Longitude (California)
}
```
The model predicts the house price based on trained parameters and displays:
```bash
Predicted House Value: $369,580.43
```

## 📌 Results & Insights
- **Polynomial Feature Expansion** improved model performance.
- **Lasso Regression** helped reduce overfitting by feature selection.
- **Ridge Regression** provided better stability with optimized `alpha`.
- **Feature Importance** analysis highlighted `MedInc` as the most influential predictor.



## 📜 License
This project is open-source and available under the **MIT License**.


## 📧 Contact

- **Email:** bilawal27saeed@gmail.com
- **GitHub:** [bilawalsaeed](https://github.com/bilawalsaeed)
