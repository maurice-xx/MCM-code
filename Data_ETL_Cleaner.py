import pandas as pd
import numpy as np
import re

# ==========================================
# 1. 初始化路径与加载数据
# ==========================================
input_file = "D:/2026_MCM_Problem_C_Data.csv"
output_path = "D:/MCM-code/"

try:
    df_raw = pd.read_csv(input_file)
    print("原始数据加载成功！")
except FileNotFoundError:
    print(f"错误：在 {input_file} 未找到文件，请检查路径。")
    exit()

# ==========================================
# 2. 生成文件 1：每周表现明细表 (Weekly_Performance.csv)
# 目的：处理裁判分，计算 Rank 和 Percent [cite: 95, 104]
# ==========================================
weekly_rows = []
# 遍历最大可能的比赛周 (1-11周) [cite: 81]
for w in range(1, 12):
    judge_cols = [c for c in df_raw.columns if f'week{w}_judge' in c]
    
    for _, row in df_raw.iterrows():
        # 获取评分并处理 N/A 值 [cite: 79]
        total_score = row[judge_cols].sum()
        
        # 只记录分数有效（大于0且非空）的选手 [cite: 82, 84]
        if pd.notna(total_score) and total_score > 0:
            weekly_rows.append({
                'Season': row['season'],
                'Celebrity': row['celebrity_name'],
                'Week': w,
                'Judge_Total': total_score,
            })

df_weekly = pd.DataFrame(weekly_rows)

# 分组计算当周裁判排名 (Rank) 和 分数百分比 (Percent) [cite: 101]
# method='min' 确保并列得分拥有相同排名 [cite: 95]
df_weekly['Judge_Rank'] = df_weekly.groupby(['Season', 'Week'])['Judge_Total'].rank(ascending=False, method='min')
df_weekly['Judge_Pct'] = df_weekly['Judge_Total'] / df_weekly.groupby(['Season', 'Week'])['Judge_Total'].transform('sum')

df_weekly.to_csv(f"{output_path}Weekly_Performance.csv", index=False)


# ==========================================
# 3. 生成文件 2：选手特征分析表 (Contestant_Features.csv)
# 目的：包含数值化的 Region、Is_AllStar 和 Partner 指数
# ==========================================
df_features = df_raw[[
    'celebrity_name', 'celebrity_industry', 'celebrity_age_during_season', 
    'ballroom_partner', 'placement', 'season', 'celebrity_homestate', 'celebrity_homecountry/region'
]].copy()

# 重命名以保持整洁 
df_features.columns = ['Name', 'Industry', 'Age', 'Partner', 'Final_Placement', 'Season', 'State', 'Country']

# [A] 标记全明星赛季 
df_features['Is_AllStar'] = (df_features['Season'] == 15).astype(int)

# [B] 处理地域映射 (Region) 
state_to_region = {
    'Alabama': 'South', 'Alaska': 'West', 'Arizona': 'West', 'Arkansas': 'South', 'California': 'West',
    'Colorado': 'West', 'Connecticut': 'Northeast', 'Delaware': 'Northeast', 'Florida': 'South',
    'Georgia': 'South', 'Hawaii': 'West', 'Idaho': 'West', 'Illinois': 'Midwest', 'Indiana': 'Midwest',
    'Iowa': 'Midwest', 'Kansas': 'Midwest', 'Kentucky': 'South', 'Louisiana': 'South', 'Maine': 'Northeast',
    'Maryland': 'Northeast', 'Massachusetts': 'Northeast', 'Michigan': 'Midwest', 'Minnesota': 'Midwest',
    'Mississippi': 'South', 'Missouri': 'Midwest', 'Montana': 'West', 'Nebraska': 'Midwest', 'Nevada': 'West',
    'New Hampshire': 'Northeast', 'New Jersey': 'Northeast', 'New Mexico': 'West', 'New York': 'Northeast',
    'North Carolina': 'South', 'North Dakota': 'Midwest', 'Ohio': 'Midwest', 'Oklahoma': 'South',
    'Oregon': 'West', 'Pennsylvania': 'Northeast', 'Rhode Island': 'Northeast', 'South Carolina': 'South',
    'South Dakota': 'Midwest', 'Tennessee': 'South', 'Texas': 'South', 'Utah': 'West', 'Vermont': 'Northeast',
    'Virginia': 'South', 'Washington': 'West', 'West Virginia': 'South', 'Wisconsin': 'Midwest', 'Wyoming': 'West'
}

def map_region(row):
    if pd.notna(row['Country']) and row['Country'] != 'United States':
        return 'International'
    state = str(row['State']).strip()
    return state_to_region.get(state, 'Other/Unknown')

df_features['Region_Cat'] = df_features.apply(map_region, axis=1)

# [C] 舞伴实力量化 (Partner_Strength_Index)
partner_avg = df_features.groupby('Partner')['Final_Placement'].mean().to_dict()
df_features['Partner_Strength_Index'] = df_features['Partner'].map(partner_avg)

# [D] 重要：将 Region 进行 One-hot 编码并整合
region_dummies = pd.get_dummies(df_features['Region_Cat'], prefix='Region')
df_features_final = pd.concat([df_features, region_dummies], axis=1)

# [E] 剔除所有不可计算的文本列，仅保留数值特征列用于机器学习
cols_to_drop = [ 'Industry', 'Partner', 'State', 'Country', 'Region_Cat']
df_features_final = df_features_final.drop(columns=cols_to_drop)

df_features_final.to_csv(f"{output_path}Contestant_Features.csv", index=False)


# ==========================================
# 4. 生成文件 3：淘汰结果对照表 (Elimination_Lookup.csv)
# 目的：提取每位选手的淘汰周作为反演依据 
# ==========================================
def extract_elim_week(res_str):
    if pd.isna(res_str): return None
    if "Place" in res_str: return 99 # 决赛选手
    nums = re.findall(r'\d+', res_str)
    return int(nums[0]) if nums else None

df_elim = df_raw[['season', 'celebrity_name', 'results']].copy()
df_elim['Elim_Week'] = df_elim['results'].apply(extract_elim_week)

df_elim.to_csv(f"{output_path}Elimination_Lookup.csv", index=False)

print("-" * 30)
print("所有处理表已成功生成至 D 盘！")
print("1. Weekly_Performance.csv (含裁判排名与占比)")
print("2. Contestant_Features.csv (含编码后的地域、全明星与舞伴指数)")
print("3. Elimination_Lookup.csv (含准确的淘汰周标签)")