import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

class FactorAnalysis:
    """分析舞者/明星特征对评委分与观众投票的影响"""
    
    def __init__(self):
        # 加载数据
        self.weekly = pd.read_csv("Weekly_Performance.csv")
        self.features = pd.read_csv("Contestant_Features.csv")
        self.estimated_fan = pd.read_csv("Estimated_Fan_Votes_Final_Model_2.csv")
        self.raw_data = pd.read_csv("2026_MCM_Problem_C_Data.csv")
        
    def extract_dancer_features(self):
        """提取舞者特征"""
        dancer_stats = {}
        
        for dancer in self.raw_data['ballroom_partner'].unique():
            if pd.isna(dancer):
                continue
            dancer_data = self.raw_data[self.raw_data['ballroom_partner'] == dancer]
            
            seasons_count = dancer_data['season'].nunique()
            avg_placement = dancer_data['placement'].mean()
            
            dancer_stats[dancer] = {
                'seasons_count': seasons_count,
                'avg_placement': avg_placement,
                'is_allstar': seasons_count >= 2,
                'experience_level': min(seasons_count, 3)
            }
        
        return pd.DataFrame(dancer_stats).T
    
    def create_merged_dataset(self):
        """合并所有数据集"""
        # 1. 从原始数据提取基础信息
        raw_info = self.raw_data[['celebrity_name', 'ballroom_partner', 
                                   'celebrity_industry', 'celebrity_age_during_season', 
                                   'season']].drop_duplicates()
        
        # 2. 合并周度评委成绩与观众投票
        merged = self.weekly.merge(
            self.estimated_fan[['Season', 'Week', 'Celebrity', 
                                 'Estimated_Fan_Pct', 'Estimated_Fan_Rank']],
            on=['Season', 'Week', 'Celebrity'],
            how='left'
        )
        
        # 3. 添加明星基础信息
        merged = merged.merge(
            raw_info,
            left_on=['Celebrity', 'Season'],
            right_on=['celebrity_name', 'season'],
            how='left'
        )
        
        # 4. 添加舞者特征
        dancer_features = self.extract_dancer_features()
        merged['Dancer'] = merged['Celebrity'].map(
            self.raw_data.set_index('celebrity_name')['ballroom_partner'].to_dict()
        )
        
        merged = merged.merge(
            dancer_features,
            left_on='Dancer',
            right_index=True,
            how='left'
        )
        
        # 5. 特征工程
        merged['Age'] = merged['celebrity_age_during_season']
        merged['Age_Squared'] = merged['Age'] ** 2
        merged['Industry'] = merged['celebrity_industry']
        merged['Judge_Fan_Diff'] = merged['Judge_Pct'] - merged['Estimated_Fan_Pct']
        
        # 计算进度周数
        merged['Week_Progress'] = merged.groupby(['Season', 'Celebrity'])['Week'].transform(
            lambda x: x / x.max()
        )
        
        # 行业编码
        industry_mapping = {ind: i for i, ind in enumerate(merged['Industry'].unique())}
        merged['Industry_Code'] = merged['Industry'].map(industry_mapping)
        
        return merged.dropna(subset=['Judge_Pct', 'Estimated_Fan_Pct', 'Age', 
                                      'Industry_Code', 'experience_level'])
    
    def analyze_judge_score_drivers(self, data):
        """评委评分驱动因素"""
        X = data[['Age', 'Age_Squared', 'is_allstar', 'experience_level', 
                  'Week_Progress', 'Industry_Code']].fillna(0)
        y = data['Judge_Pct']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf_judge = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_judge.fit(X_scaled, y)
        
        importance_judge = pd.DataFrame({
            'Feature': ['Age', 'Age²', 'AllStar', 'Dancer_Experience', 'Week_Progress', 'Industry'],
            'Importance': rf_judge.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return rf_judge, importance_judge
    
    def analyze_fan_vote_drivers(self, data):
        """观众投票驱动因素"""
        X = data[['Age', 'Age_Squared', 'is_allstar', 'experience_level', 
                  'Week_Progress', 'Industry_Code']].fillna(0)
        y = data['Estimated_Fan_Pct']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf_fan = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_fan.fit(X_scaled, y)
        
        importance_fan = pd.DataFrame({
            'Feature': ['Age', 'Age²', 'AllStar', 'Dancer_Experience', 'Week_Progress', 'Industry'],
            'Importance': rf_fan.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return rf_fan, importance_fan
    
    def consistency_analysis(self, data):
        """一致性分析"""
        results = {}
        
        # 年龄效应
        judge_age_corr, judge_age_p = spearmanr(data['Age'].fillna(0), data['Judge_Pct'])
        fan_age_corr, fan_age_p = spearmanr(data['Age'].fillna(0), data['Estimated_Fan_Pct'])
        
        results['Age_Effect'] = {
            'Judge_Corr': judge_age_corr,
            'Judge_Pval': judge_age_p,
            'Fan_Corr': fan_age_corr,
            'Fan_Pval': fan_age_p,
            'Consistency': 'YES' if np.sign(judge_age_corr) == np.sign(fan_age_corr) else 'NO'
        }
        
        # AllStar 效应
        allstar_data = data.groupby('is_allstar').agg({
            'Judge_Pct': 'mean',
            'Estimated_Fan_Pct': 'mean'
        })
        
        if True in allstar_data.index and False in allstar_data.index:
            judge_diff = allstar_data.loc[True, 'Judge_Pct'] - allstar_data.loc[False, 'Judge_Pct']
            fan_diff = allstar_data.loc[True, 'Estimated_Fan_Pct'] - allstar_data.loc[False, 'Estimated_Fan_Pct']
            
            results['AllStar_Effect'] = {
                'Judge_Bonus': judge_diff,
                'Fan_Bonus': fan_diff,
                'Consistency': 'YES' if np.sign(judge_diff) == np.sign(fan_diff) else 'NO'
            }
        
        # 舞者经验效应
        judge_dancer, _ = spearmanr(data['experience_level'].fillna(0), data['Judge_Pct'])
        fan_dancer, _ = spearmanr(data['experience_level'].fillna(0), data['Estimated_Fan_Pct'])
        
        results['Dancer_Experience'] = {
            'Judge_Corr': judge_dancer,
            'Fan_Corr': fan_dancer,
            'Consistency': 'YES' if np.sign(judge_dancer) == np.sign(fan_dancer) else 'NO'
        }
        
        return results
    
    def plot_comparison(self, importance_judge, importance_fan):
        """绘制对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        importance_judge.sort_values('Importance').plot(
            x='Feature', y='Importance', kind='barh', ax=axes[0], color='steelblue', legend=False
        )
        axes[0].set_title('Judge Score - Feature Importance', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Importance')
        
        importance_fan.sort_values('Importance').plot(
            x='Feature', y='Importance', kind='barh', ax=axes[1], color='coral', legend=False
        )
        axes[1].set_title('Fan Score - Feature Importance', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('03_Factor_Comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 已保存: 09_Factor_Comparison.png")
        plt.close()
    
    def generate_report(self):
        """生成分析报告"""
        print("=" * 100)
        print("第三题：特征影响因素分析")
        print("=" * 100)
        
        data = self.create_merged_dataset()
        print(f"\n✓ 数据合并完成：{len(data)} 条记录")
        
        # 评委模型
        print("\n" + "-" * 100)
        print("【模型1】评委评分的驱动因素")
        print("-" * 100)
        rf_judge, importance_judge = self.analyze_judge_score_drivers(data)
        print(importance_judge.to_string(index=False))
        
        # 观众模型
        print("\n" + "-" * 100)
        print("【模型2】观众投票的驱动因素")
        print("-" * 100)
        rf_fan, importance_fan = self.analyze_fan_vote_drivers(data)
        print(importance_fan.to_string(index=False))
        
        # 一致性分析
        print("\n" + "-" * 100)
        print("【一致性分析】评委 vs 观众")
        print("-" * 100)
        consistency = self.consistency_analysis(data)
        
        for factor, result in consistency.items():
            print(f"\n{factor}:")
            for key, val in result.items():
                print(f"  {key}: {val}")
        
        self.plot_comparison(importance_judge, importance_fan)
        
        print("\n" + "=" * 100)
        print("【分析完成】")
        print("=" * 100)

if __name__ == '__main__':
    analyzer = FactorAnalysis()
    analyzer.generate_report()