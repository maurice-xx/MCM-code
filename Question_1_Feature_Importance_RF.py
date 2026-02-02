import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

<<<<<<< HEAD
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
=======
# 设置中文显示（防止绘图中文乱码，视系统环境而定）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载特征表
df_features = pd.read_csv("D:/Contestant_Features.csv")

# 2. 准备自变量 X 和因变量 y
# 我们预测的是 Final_Placement（名次），数值越小表示表现越好
# 剔除 Season（赛季号），因为它不是选手的特征
X = df_features.drop(columns=['Final_Placement', 'Season'])
y = df_features['Final_Placement']

# 3. 构建并训练随机森林模型
# n_estimators: 森林中树的数量；random_state: 保证结果可复现
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# 4. 提取特征重要性
feature_importance = pd.DataFrame({
    '特征名称': X.columns,
    '重要性权重': rf_model.feature_importances_
}).sort_values(by='重要性权重', ascending=False)

# 5. 可视化结果
plt.figure(figsize=(10, 6))
sns.barplot(x='重要性权重', y='特征名称', data=feature_importance, palette='viridis')
plt.title("各特征对选手最终名次的影响力排名")
plt.tight_layout()
plt.show()

print("特征重要性分析完成！你可以将生成的图表放入论文中。")
# ==============================
# 6. 保存特征重要性（给 p3 用）
>>>>>>> a836be3130d3e60cb0c9084a1141fe15be639faa
# ==============================

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Weight': rf_model.feature_importances_
})

<<<<<<< HEAD
# Normalize (RF already sums to 1, but this ensures clean formatting)
=======
# 归一化（虽然 RF 已经是 sum=1，这步是为了论文好看）
>>>>>>> a836be3130d3e60cb0c9084a1141fe15be639faa
feature_importance_df['Weight'] = (
    feature_importance_df['Weight'] /
    feature_importance_df['Weight'].sum()
)

<<<<<<< HEAD
# Sort by importance
=======
# 按重要性排序
>>>>>>> a836be3130d3e60cb0c9084a1141fe15be639faa
feature_importance_df = feature_importance_df.sort_values(
    by='Weight', ascending=False
)

<<<<<<< HEAD
# Save to CSV
=======
# 保存
>>>>>>> a836be3130d3e60cb0c9084a1141fe15be639faa
feature_importance_df.to_csv(
    "D:/Feature_Importance.csv",
    index=False,
    encoding='utf-8-sig'
)

<<<<<<< HEAD
print("Feature_Importance.csv generated (for p3 inversion).")
=======
print("已生成 Feature_Importance.csv（供 p3 反演使用）")
>>>>>>> a836be3130d3e60cb0c9084a1141fe15be639faa
