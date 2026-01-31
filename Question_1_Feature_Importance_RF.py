import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

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
# ==============================

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Weight': rf_model.feature_importances_
})

# 归一化（虽然 RF 已经是 sum=1，这步是为了论文好看）
feature_importance_df['Weight'] = (
    feature_importance_df['Weight'] /
    feature_importance_df['Weight'].sum()
)

# 按重要性排序
feature_importance_df = feature_importance_df.sort_values(
    by='Weight', ascending=False
)

# 保存
feature_importance_df.to_csv(
    "D:/Feature_Importance.csv",
    index=False,
    encoding='utf-8-sig'
)

print("已生成 Feature_Importance.csv（供 p3 反演使用）")
