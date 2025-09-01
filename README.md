# Simulated Nigerian Soil and Crop Yield Dataset (SNSCYD) – Analysis Report

## Overview

This project explores a simulated agricultural dataset inspired by the National Farmers Soil Health Card Scheme (NFSHS) under the Federal Ministry of Agriculture and Food Security, Nigeria.

The dataset models key soil health indicators across 8 Nigerian states and their impact on crop yield. The goal is to understand how different soil properties affect crop productivity and to provide actionable recommendations for farmers and policymakers.

## Dataset Description

**Dataset Name**: Simulated Nigerian Soil and Crop Yield Dataset (SNSCYD)

**States Included**: Gombe, Kano, Jigawa, Sokoto, Kaduna, Benue, Oyo, Cross River  
**Number of Records**: 200 rows

### Features:

| Column Name              | Description                              |
|--------------------------|------------------------------------------|
| State                    | Nigerian state where the sample is from  |
| Soil_pH                  | Acidity/alkalinity of soil (optimal ≈ 6.5) |
| Nitrogen_kg_ha           | Nitrogen content in soil (kg/ha)         |
| Phosphorus_kg_ha         | Phosphorus content (kg/ha)               |
| Potassium_kg_ha          | Potassium content (kg/ha)                |
| Organic_Carbon_percent   | Organic carbon content in soil (%)       |
| Yield_tonnes_per_ha      | Simulated crop yield (tonnes/ha)         |

## Data Cleaning

- Removed duplicates  
- Verified data types  
- Detected outliers using IQR  
- Confirmed no missing values

## Exploratory Data Analysis (EDA)

**Key Insights:**
- Yield is positively correlated with nitrogen, phosphorus, and organic carbon.
- Organic carbon has the strongest influence.
- Soil pH that deviates from ~6.5 reduces yield.

## Regression Analysis

### Model Performance:
- R² Score: ~0.85
- MSE: Low, indicating good prediction accuracy

### Feature Impact:

| Feature               | Coefficient | Interpretation                            |
|-----------------------|-------------|--------------------------------------------|
| Nitrogen_kg_ha        | +0.20       | Every 1 kg adds ~0.20 tonnes yield         |
| Phosphorus_kg_ha      | +0.30       | Stronger yield boost                       |
| Potassium_kg_ha       | +0.10       | Smaller positive effect                    |
| Organic_Carbon_percent| +2.00       | Most important: 1% adds 2 tonnes yield     |
| Soil_pH (Δ from 6.5)  | -1.50       | Deviation reduces yield                    |

## Interpretation in Layman's Terms

More nitrogen, phosphorus, and organic material leads to more food.  
Keep soil pH close to 6.5 to maximize yield.  
Organic matter (compost, manure) plays a large role in improving yield.

## Recommendations

### For Farmers:
- Use soil test kits
- Add compost or manure
- Apply balanced NPK fertilizers
- Adjust pH using lime or sulfur if needed

### For Government:
- Expand access to soil testing (NFSHS)
- Support composting initiatives
- Train farmers on pH balancing

### For Researchers:
- Apply model to real NFSHS data
- Use machine learning for deeper insight

## Project Files

- `snsyd_simulation.ipynb` – Code and analysis
- `README.md` – This documentation
- `snsyd_cleaned.csv` – Clean dataset (optional)

## Acknowledgments

Inspired by the Federal Ministry of Agriculture and Food Security – https://agriculture.gov.ng/
