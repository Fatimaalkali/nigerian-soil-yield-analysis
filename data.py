import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

states = ['Gombe', 'Kano', 'Jigawa', 'Sokoto', 'Kaduna', 'Benue', 'Oyo', 'Cross River']
n_samples = 200

state_data = np.random.choice(states, size=n_samples)

soil_pH = np.round(np.random.normal(loc=6.5, scale=0.5, size=n_samples), 2)
nitrogen = np.random.randint(50, 150, size=n_samples)  # kg/ha
phosphorus = np.random.randint(10, 60, size=n_samples)  # kg/ha
potassium = np.random.randint(100, 250, size=n_samples)  # kg/ha
organic_carbon = np.round(np.random.uniform(0.5, 3.0, size=n_samples), 2)  # %

yield_tonnes_per_ha = (
    0.2 * nitrogen +
    0.3 * phosphorus +
    0.1 * potassium +
    2.0 * organic_carbon -
    1.5 * abs(soil_pH - 6.5) +  # Optimal pH is 6.5
    np.random.normal(0, 3, size=n_samples)  # add some noise
)

# Create DataFrame
df = pd.DataFrame({
    'State': state_data,
    'Soil_pH': soil_pH,
    'Nitrogen_kg_ha': nitrogen,
    'Phosphorus_kg_ha': phosphorus,
    'Potassium_kg_ha': potassium,
    'Organic_Carbon_percent': organic_carbon,
    'Yield_tonnes_per_ha': np.round(yield_tonnes_per_ha, 2)
})

print(df.head())

#Data cleaning
print("Missing values:\n", df.isnull().sum())

print("\nData types:\n", df.dtypes)

print("\nSummary statistics:\n", df.describe())

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

df = df.drop_duplicates()

numeric_cols = ['Soil_pH', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha', 'Organic_Carbon_percent', 'Yield_tonnes_per_ha']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")

print(f"\nCleaned dataset shape: {df.shape}")


#data cleaning
print("Missing values:\n", df.isnull().sum())

print("\nData types:\n", df.dtypes)

print("\nSummary statistics:\n", df.describe())

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

df = df.drop_duplicates()

numeric_cols = ['Soil_pH', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha', 'Potassium_kg_ha', 'Organic_Carbon_percent', 'Yield_tonnes_per_ha']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")

print(f"\nCleaned dataset shape: {df.shape}")

#EDA
sns.set(style='whitegrid')

df.hist(figsize=(12, 8), bins=20)
plt.suptitle("Distribution of Soil Properties and Yield", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='State', y='Yield_tonnes_per_ha', data=df)
plt.title("Crop Yield Distribution by State")
plt.xticks(rotation=45)
plt.ylabel("Yield (tonnes/ha)")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(x='Nitrogen_kg_ha', y='Yield_tonnes_per_ha', data=df, ax=axes[0, 0])
axes[0, 0].set_title("Yield vs Nitrogen")

sns.scatterplot(x='Phosphorus_kg_ha', y='Yield_tonnes_per_ha', data=df, ax=axes[0, 1])
axes[0, 1].set_title("Yield vs Phosphorus")

sns.scatterplot(x='Potassium_kg_ha', y='Yield_tonnes_per_ha', data=df, ax=axes[1, 0])
axes[1, 0].set_title("Yield vs Potassium")

sns.scatterplot(x='Organic_Carbon_percent', y='Yield_tonnes_per_ha', data=df, ax=axes[1, 1])
axes[1, 1].set_title("Yield vs Organic Carbon")

plt.tight_layout()
plt.show()

#REGRESSION
features = ['Soil_pH', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha',
            'Potassium_kg_ha', 'Organic_Carbon_percent']
target = 'Yield_tonnes_per_ha'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("📈 Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print("\n📊 Feature Impact on Yield:")
print(coefficients)
