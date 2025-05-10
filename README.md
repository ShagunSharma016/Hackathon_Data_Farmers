# 🌾 Smarter Farming with Data: Crop Yield Analysis & Prediction

> 🚀 A data-driven solution to help farmers and policymakers improve crop productivity using machine learning and visual analytics.

---

## 📌 Why This Project?

India is an agricultural powerhouse, yet many farmers still rely on intuition over data.  
This project brings together **insights + predictions** to:

- 📊 Understand how inputs like rainfall, fertilizer, and season affect yield
- 🤖 Predict crop yield accurately using ML
- 🌐 Make smarter, region-specific farming decisions

---

## 📂 What’s Inside?

| Module | Purpose |
|--------|---------|
| 🔍 **Power BI Dashboards** | Visualize trends, patterns, and comparisons |
| 🤖 **ML Model (Random Forest)** | Predict crop yield from input features |
| 📈 **Data Insights** | Extract actionable conclusions from raw data |

---

## 🧠 Core Questions We Answer

- Which **states and crops** produce the best yields?
- Does **more rainfall = better yield**? What about fertilizer?
- Can we **predict yield** before harvesting?

---

## 📊 Visual Analysis with Power BI

Power BI dashboards helped break down thousands of crop records into easy, insightful visuals.

### 🔍 Highlights:

- 🥇 **Top Performing Crops**: Rice, Maize, Moong
- 🗺️ **Best States by Yield**: Punjab, Andhra Pradesh, Tamil Nadu
- 🌧️ **Rainfall vs Yield**: Clear positive trend
- 🧪 **Input Correlation**: Fertilizer use shows strong impact

> 📌 *Visuals created from `Hackthon.pbix`*

---

## 🤖 Machine Learning: Predicting Crop Yield

Used `RandomForestRegressor` to predict `Yield (kg/hectare)` from inputs like:

- Rainfall
- Fertilizer and Pesticide per hectare
- Crop, Season, State (encoded)
- Area

### ✅ Machine Learning Code

```python
# 1. Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 2. Loading the dataset
data = pd.read_excel('crop_cleaned.xlsx')  # Change this path to your file location

# 3. Droping unnecessary columns
data = data.drop(columns=['Crop_Year'])

# 4. Define features and target
X = data.drop(columns=['Yield'])
y = data['Yield']

# 5. Define categorical and numerical columns
categorical_features = ['Crop', 'Season', 'State']
numerical_features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# 6. Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 7. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
model.fit(X_train, y_train)

# 9. Test the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R2 Score on test set: {r2}')

# 10. Predict for new input (example: Onion in Assam)
# input_data = pd.DataFrame({
#     'Area': [6637],
#     'Production': [135487.22],
#     'Annual_Rainfall': [2051.4],
#     'Fertilizer': [95.17],
#     'Pesticide': [0.31],
#     'Crop': ['Onion'],
#     'Season': ['Whole Year'],
#     'State': ['Assam']
# })

# input_data = pd.DataFrame({
#     'Area': [5637],
#     'Production': [95454.22],
#     'Annual_Rainfall': [2800.4],
#     'Fertilizer': [87.17],
#     'Pesticide': [0.29],
#     'Crop': ['Wheat'],
#     'Season': ['Whole Year'],
#     'State': ['Karnataka']
# })

input_data = pd.DataFrame({
    'Area': [5000],
    'Production': [120000],
    'Annual_Rainfall': [950],
    'Fertilizer': [85],
    'Pesticide': [0.45],
    'Crop': ['Maize'],
    'Season': ['Kharif'],
    'State': ['Karnataka']
})
```
# 11. Predict yield
predicted_yield = model.predict(input_data)
print(f'Predicted Yield: {predicted_yield[0]}')

---

## ✅ Conclusion

- 🌿 This project bridges the gap between agriculture and data science, turning raw farming data into actionable insights.
- 📊 With the Power BI dashboard, we uncovered hidden trends and state-wise patterns in crop yield performance.
- 🤖 Using machine learning, we built a reliable yield prediction model**, empowering farmers and policymakers to make data-informed decisions.
- 🔁 The combination of  analysis + prediction** makes this project highly scalable — ready for integration into  real-world agriculture systems or smart farming apps.

> 💡 In short:  Data can grow crops smarter — and this project proves it. 🌾
