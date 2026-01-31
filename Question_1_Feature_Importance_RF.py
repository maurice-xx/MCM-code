import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Set font for English display (default matplotlib fonts work well for English)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 1. Load feature data
df_features = pd.read_csv("D:/Contestant_Features.csv")

# 2. Prepare independent variables X and dependent variable y
# We predict Final_Placement (lower value = better performance)
# Drop Season as it's not a contestant characteristic
X = df_features.drop(columns=['Final_Placement', 'Season'])
y = df_features['Final_Placement']

# 3. Build and train Random Forest model
# n_estimators: number of trees; random_state: reproducibility
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# 4. Extract feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 5. Visualization (English labels)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title("Feature Importance Ranking for Final Placement", fontsize=14, fontweight='bold')
plt.xlabel("Importance Weight", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)
plt.tight_layout()
plt.show()

print("Feature importance analysis completed! You can include this chart in your thesis.")

# ==============================
# 6. Save feature importance (for p3 inversion)
# ==============================

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Weight': rf_model.feature_importances_
})

# Normalize (RF already sums to 1, but this ensures clean formatting)
feature_importance_df['Weight'] = (
    feature_importance_df['Weight'] /
    feature_importance_df['Weight'].sum()
)

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by='Weight', ascending=False
)

# Save to CSV
feature_importance_df.to_csv(
    "D:/Feature_Importance.csv",
    index=False,
    encoding='utf-8-sig'
)

print("Feature_Importance.csv generated (for p3 inversion).")