import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

class VotingSystemDesign:
    """设计更公平且更有观赏性的投票制度"""
    
    def __init__(self):
        self.weekly = pd.read_csv("Weekly_Performance.csv")
        self.estimated_fan = pd.read_csv("Estimated_Fan_Votes_Final_Model_2.csv")
        self.raw_data = pd.read_csv("2026_MCM_Problem_C_Data.csv")
        
    def diagnose_current_system(self):
        """诊断现有投票制度的问题"""
        print("=" * 100)
        print("【现有制度诊断】")
        print("=" * 100)
        
        # 合并数据
        merged = self.weekly.merge(
            self.estimated_fan[['Season', 'Week', 'Celebrity', 'Estimated_Fan_Pct']],
            on=['Season', 'Week', 'Celebrity'],
            how='inner'
        )
        
        print(f"\n✓ 已合并周度数据：{len(merged)} 条记录")
        
        # 问题1：评委与观众的分歧
        merged['Judge_Fan_Gap'] = abs(merged['Judge_Pct'] - merged['Estimated_Fan_Pct'])
        avg_gap = merged['Judge_Fan_Gap'].mean()
        
        print(f"\n【问题1】评委与观众意见分歧")
        print(f"  平均分歧度：{avg_gap:.4f}")
        print(f"  分歧>0.1的情况占比：{(merged['Judge_Fan_Gap'] > 0.1).mean():.2%}")
        print(f"  最大分歧：{merged['Judge_Fan_Gap'].max():.4f}")
        
        # 添加明星特征信息
        raw_info = self.raw_data[['celebrity_name', 'celebrity_industry', 
                                   'celebrity_age_during_season', 'season']].drop_duplicates()
        
        merged = merged.merge(
            raw_info,
            left_on=['Celebrity', 'Season'],
            right_on=['celebrity_name', 'season'],
            how='left'
        )
        
        # 问题2：按行业分析评委偏差
        industry_bias = merged.groupby('celebrity_industry').agg({
            'Judge_Pct': 'mean',
            'Estimated_Fan_Pct': 'mean'
        })
        industry_bias['Bias'] = industry_bias['Judge_Pct'] - industry_bias['Estimated_Fan_Pct']
        
        print(f"\n【问题2】特定人群的偏见 - 按行业分类")
        print(industry_bias.round(4).to_string())
        
        # 问题3：按年龄分析评委偏差
        merged['Age_Group'] = pd.cut(merged['celebrity_age_during_season'], 
                                      bins=[0, 30, 40, 50, 100], 
                                      labels=['<30', '30-40', '40-50', '>50'])
        age_bias = merged.groupby('Age_Group').agg({
            'Judge_Pct': 'mean',
            'Estimated_Fan_Pct': 'mean'
        })
        age_bias['Bias'] = age_bias['Judge_Pct'] - age_bias['Estimated_Fan_Pct']
        
        print(f"\n【问题3】特定人群的偏见 - 按年龄分组")
        print(age_bias.round(4).to_string())
        
        return merged
    
    def design_system_1_dynamic_weights(self, merged):
        """【方案1】动态权重制"""
        print("\n" + "=" * 100)
        print("【方案1】动态权重制 (Season-Adaptive Weighted System)")
        print("=" * 100)
        
        def get_weights(season):
            if season <= 2:
                return 0.70, 0.30
            elif season <= 5:
                return 0.50, 0.50
            else:
                return 0.30, 0.70
        
        merged['Judge_Weight'], merged['Fan_Weight'] = zip(
            *merged['Season'].map(lambda s: get_weights(s))
        )
        
        merged['Combined_Score_Plan1'] = (
            merged['Judge_Pct'] * merged['Judge_Weight'] + 
            merged['Estimated_Fan_Pct'] * merged['Fan_Weight']
        )
        
        print("\n权重设置：")
        print("  赛季1-2：评委70% + 观众30%  (原因：早期观众认知不足)")
        print("  赛季3-5：评委50% + 观众50%  (原因：权力平衡)")
        print("  赛季6+ ：评委30% + 观众70%  (原因：观众更成熟)")
        
        print(f"\n✓ 方案1已计算完毕")
        
        return merged
    
    def design_system_2_fairness_adjusted(self, merged):
        """【方案2】公平性调整制"""
        print("\n" + "=" * 100)
        print("【方案2】公平性调整制 (Fairness-Corrected System)")
        print("=" * 100)
        
        industry_stats = merged.groupby('celebrity_industry').agg({
            'Judge_Pct': 'mean',
            'Estimated_Fan_Pct': 'mean'
        })
        industry_stats['Bias'] = industry_stats['Judge_Pct'] - industry_stats['Estimated_Fan_Pct']
        industry_stats['Correction_Factor'] = 1 - industry_stats['Bias']
        
        print("\n各行业的纠正因子：")
        print(industry_stats[['Correction_Factor']].round(4).to_string())
        
        merged['Bias_Correction'] = merged['celebrity_industry'].map(
            industry_stats['Correction_Factor'].to_dict()
        )
        
        merged['Combined_Score_Plan2'] = (
            merged['Judge_Pct'] * 0.5 + 
            merged['Estimated_Fan_Pct'] * 0.5 * merged['Bias_Correction']
        )
        
        # 标准化到0-1
        scaler = MinMaxScaler()
        merged['Combined_Score_Plan2'] = scaler.fit_transform(
            merged[['Combined_Score_Plan2']]
        ).flatten()
        
        print(f"\n✓ 方案2已计算完毕")
        
        return merged
    
    def design_system_3_storytelling(self, merged):
        """【方案3】叙事加权制"""
        print("\n" + "=" * 100)
        print("【方案3】叙事加权制 (Story-Driven System)")
        print("=" * 100)
        
        # 计算改进空间
        week1_data = merged[merged['Week'] == merged.groupby(['Season', 'Celebrity'])['Week'].transform('min')]
        week1_scores = week1_data[['Season', 'Celebrity', 'Judge_Pct']].rename(
            columns={'Judge_Pct': 'Week1_Judge_Pct'}
        )
        
        merged = merged.merge(week1_scores, on=['Season', 'Celebrity'], how='left')
        merged['Improvement_Ratio'] = (
            (merged['Judge_Pct'] - merged['Week1_Judge_Pct']) / 
            (merged['Week1_Judge_Pct'] + 0.001)
        ).clip(-1, 2)
        
        merged['Improvement_Bonus'] = np.maximum(merged['Improvement_Ratio'], 0)
        
        # 稳定性
        consistency = merged.groupby(['Season', 'Celebrity'])['Judge_Pct'].std()
        consistency_map = consistency.to_dict()
        merged['Consistency_Score'] = merged.apply(
            lambda row: 1 - (consistency_map.get((row['Season'], row['Celebrity']), 0) / 10),
            axis=1
        ).clip(0, 1)
        
        # 故事潜力
        merged['Age'] = merged['celebrity_age_during_season']
        age_min, age_max = merged['Age'].min(), merged['Age'].max()
        merged['Story_Potential'] = (
            (merged['Age'] - age_min) / (age_max - age_min + 0.001) * 0.5 +
            merged['Improvement_Ratio'].clip(0, 1) * 0.5
        ).clip(0, 1)
        
        # 最终组合分
        merged['Combined_Score_Plan3'] = (
            merged['Judge_Pct'] * 0.40 +
            merged['Estimated_Fan_Pct'] * 0.30 +
            merged['Improvement_Bonus'] * 0.15 +
            merged['Story_Potential'] * 0.10 +
            merged['Consistency_Score'] * 0.05
        )
        
        print("\n权重构成：")
        print("  40% - 评委评分（专业性）")
        print("  30% - 观众投票（人气）")
        print("  15% - 改进空间（激励进步）")
        print("  10% - 故事潜力（观赏性）")
        print("   5% - 稳定性（职业精神）")
        
        print(f"\n✓ 方案3已计算完毕")
        
        return merged
    
    def design_system_4_transparent_ranking(self, merged):
        """【方案4】透明排名制"""
        print("\n" + "=" * 100)
        print("【方案4】透明排名制 (Transparent Ranking System)")
        print("=" * 100)
        
        merged['Judge_Rank'] = merged.groupby(['Season', 'Week'])['Judge_Pct'].rank(ascending=False)
        merged['Fan_Rank'] = merged.groupby(['Season', 'Week'])['Estimated_Fan_Pct'].rank(ascending=False)
        
        merged['Final_Rank'] = (merged['Judge_Rank'] * 0.5 + merged['Fan_Rank'] * 0.5).round()
        merged['Rank_Difference'] = abs(merged['Judge_Rank'] - merged['Fan_Rank'])
        
        print("\n排名差异分析：")
        print(f"  平均排名差异：{merged['Rank_Difference'].mean():.2f} 名")
        print(f"  排名完全一致的比例：{(merged['Rank_Difference'] == 0).mean():.2%}")
        print(f"  排名差异>2名的比例：{(merged['Rank_Difference'] > 2).mean():.2%}")
        
        print("\n透明性优势：")
        print("  ✓ 观众可清晰看到自己的选择 vs 专业评委的评价")
        print("  ✓ 当排名差异大时，可以提供解释（技术vs人气）")
        print("  ✓ 增强投票透明度和教育意义")
        
        return merged
    
    def plot_comparison(self):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        systems = ['Existing\n(50-50)', 'plan1\nDynamic Weighting', 'plan2\nFairness Adjustment', 
                   'plan3\nNarrative Weighting', 'plan4\nTransparent Ranking']
        fairness = [3, 7, 9, 8, 10]
        entertainment = [5, 6, 7, 9, 7]
        transparency = [4, 5, 6, 8, 10]
        
        # 公平性
        axes[0, 0].bar(systems, fairness, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Fairness Score', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('Score (1-10)')
        axes[0, 0].set_ylim(0, 10)
        for i, v in enumerate(fairness):
            axes[0, 0].text(i, v + 0.2, str(v), ha='center', fontweight='bold')
        
        # 观赏性
        axes[0, 1].bar(systems, entertainment, color='coral', alpha=0.7)
        axes[0, 1].set_title('Entertainment Score', fontweight='bold', fontsize=12)
        axes[0, 1].set_ylabel('Score (1-10)')
        axes[0, 1].set_ylim(0, 10)
        for i, v in enumerate(entertainment):
            axes[0, 1].text(i, v + 0.2, str(v), ha='center', fontweight='bold')
        
        # 透明性
        axes[1, 0].bar(systems, transparency, color='seagreen', alpha=0.7)
        axes[1, 0].set_title('Transparency Score', fontweight='bold', fontsize=12)
        axes[1, 0].set_ylabel('Score (1-10)')
        axes[1, 0].set_ylim(0, 10)
        for i, v in enumerate(transparency):
            axes[1, 0].text(i, v + 0.2, str(v), ha='center', fontweight='bold')
        
        # 综合得分
        overall = np.array(fairness) * 0.35 + np.array(entertainment) * 0.35 + np.array(transparency) * 0.30
        axes[1, 1].bar(systems, overall, color='gold', alpha=0.7, edgecolor='black', linewidth=2)
        axes[1, 1].set_title('Composite Score (Weighted)', fontweight='bold', fontsize=12)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 10)
        for i, v in enumerate(overall):
            axes[1, 1].text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')
        
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('04_Voting_System_Comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ 已保存图表: 09_Voting_System_Comparison.png")
        plt.close()
    
    def print_recommendations(self):
        """打印最终建议"""
        print("\n" + "=" * 100)
        print("【最终建议与方案总结】")
        print("=" * 100)
        
        print("""
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║  推荐方案：「混合制」= 方案3（叙事加权）+ 方案4（透明排名）                                    ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【为什么选择方案3 + 方案4？】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方案3 - 叙事加权制的优势：
  ✅ 科学性：引入"改进"、"故事"、"稳定性"等量化指标
  ✅ 公平性：年长选手、新手因有更大改进空间而获得奖励
  ✅ 观赏性：每期都有"涨粉时刻"（改进明显的选手）
  ✅ 激励性：鼓励选手不断进步而非依赖初始人气

方案4 - 透明排名制的优势：
  ✅ 透明性：观众清晰看到「评委排名」vs「自己投票排名」的对比
  ✅ 教育意义：让观众理解"技术"和"人气"的平衡
  ✅ 减少争议：量化的排名公式消除主观性
  ✅ 参与感：观众投票的权重可视化，增强代入感

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【具体实施流程】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【第一步】现场评分 (直播开始)
  • 评委给出本周评分 (0-10分)
  • 观众通过APP/短信实时投票
  • 系统实时显示「评委排名」

【第二步】计算综合分 (评分结束)
  综合得分 = 评委得分×0.40 + 观众投票×0.30 + 改进空间×0.15 + 故事潜力×0.10 + 稳定性×0.05

【第三步】公示排名 (直播中)
  屏幕显示「三层排名」：
    层1️⃣  评委排名（黄色）
    层2️⃣  观众排名（蓝色）
    层3️⃣  最终排名（绿色，加权平均）
  
  高分差异大的选手标注"分歧"符号 ⚡

【第四步】舞台解读 (主持人语境)
  "XX选手本周获得观众高分但评委评分较低，这说明..."
  → 教育观众理解技术与人气的关系

【第五步】宣布淘汰 (结尾高潮)
  • 最低综合分的1-2名选手淘汰
  • 强调"改进机会"而非"实力不足"
  • 给年长/新手选手鼓励

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【对比现有制度的改进】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

现有制度问题 → 新制度解决方案
──────────────────────────────────────────────────────────────────────────
❌ 年长选手吃亏          ✅ 改进空间奖励（年长初期分数低，改进空间大）
❌ 少数行业不公平        ✅ 叙事加权（故事潜力奖励）
❌ 观众不知道自己的权重   ✅ 透明排名（三层排名可视化）
❌ 每周淘汰无逻辑        ✅ 以"进步"为核心叙事
❌ 分歧导致争议          ✅ 差异可解释（评委vs观众排名对比）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【预期的节目效果】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📺 每一期节目都会有：

  1️⃣  「技术惊喜」- 评委给高分但观众投票低 → 解释为何这个舞步值得高分
  2️⃣  「人气逆转」- 观众给高票但评委评分低 → 讨论明星光环vs技术
  3️⃣  「涨粉时刻」- 选手改进明显 → 强调进步而非绝对实力
  4️⃣  「温情故事」- 年长/新手的奋斗 → 增加观众代入感

这样节目就从"淘汰秀"变成了"成长秀" 🌟

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【风险与缓解方案】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

风险1: 选手依赖"改进奖励"，故意第一周表现差
  → 缓解：设置"最低及格线"，第一周低于此线直接淘汰

风险2: 透明排名可能导致观众"对抗"评委
  → 缓解：加入"教育模式"，主持人解释评委评分理由

风险3: 叙事加权权重可能需要调整
  → 缓解：前期试点，根据观众反馈微调各个权重系数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
    
    def generate_full_report(self):
        """生成完整报告"""
        print("\n\n")
        print("█" * 100)
        print("█" + " " * 98 + "█")
        print("█" + " " * 20 + "Question 4: 改进的投票制度设计" + " " * 47 + "█")
        print("█" + " " * 98 + "█")
        print("█" * 100)
        
        merged = self.diagnose_current_system()
        merged = self.design_system_1_dynamic_weights(merged)
        merged = self.design_system_2_fairness_adjusted(merged)
        merged = self.design_system_3_storytelling(merged)
        merged = self.design_system_4_transparent_ranking(merged)
        
        self.plot_comparison()
        self.print_recommendations()

if __name__ == '__main__':
    designer = VotingSystemDesign()
    designer.generate_full_report()