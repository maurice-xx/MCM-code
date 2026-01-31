import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

class 方法对比分析:
    """对比排名法和百分比法两种淘汰方式"""
    
    def __init__(self):
        # 加载所有需要的数据文件
        self.周度成绩 = pd.read_csv("Weekly_Performance.csv")
        self.淘汰记录 = pd.read_csv("Elimination_Lookup.csv")
        self.估计粉丝投票 = pd.read_csv("Estimated_Fan_Votes_Final_Model_2.csv")
        self.选手特征 = pd.read_csv("Contestant_Features.csv")
        
        # 添加规则类型标记
        self.周度成绩['规则'] = self.周度成绩['Season'].apply(
            lambda s: "排名法" if s <= 2 else "百分比法"
        )
        self.估计粉丝投票['规则'] = self.估计粉丝投票['Season'].apply(
            lambda s: "排名法" if s <= 2 else "百分比法"
        )
    
    def 获取评委成绩(self, season, week):
        """获取指定赛季和周次的评委成绩"""
        周数据 = self.周度成绩[
            (self.周度成绩['Season'] == season) & 
            (self.周度成绩['Week'] == week)
        ]
        return 周数据
    
    def 获取粉丝投票(self, season, week):
        """获取指定赛季和周次的估计粉丝投票"""
        周数据 = self.估计粉丝投票[
            (self.估计粉丝投票['Season'] == season) & 
            (self.估计粉丝投票['Week'] == week)
        ]
        return 周数据
    
    def 排名法淘汰(self, season, week):
        """
        排名法淘汰规则:
        评委排名 + 粉丝排名，总和最高的被淘汰
        """
        评委数据 = self.获取评委成绩(season, week)
        粉丝数据 = self.获取粉丝投票(season, week)
        
        if len(评委数据) == 0 or len(粉丝数据) == 0:
            return None, None
        
        结果列表 = []
        for _, 粉丝行 in 粉丝数据.iterrows():
            名人 = 粉丝行['Celebrity']
            评委行 = 评委数据[评委数据['Celebrity'] == 名人]
            
            if len(评委行) > 0:
                评委排名 = 评委行.iloc[0]['Judge_Rank']
                粉丝排名 = 粉丝行['Estimated_Fan_Rank']
                
                if pd.notna(评委排名) and pd.notna(粉丝排名):
                    综合排名 = 评委排名 + 粉丝排名
                    结果列表.append({
                        'Celebrity': 名人,
                        'Judge_Rank': 评委排名,
                        'Fan_Rank': 粉丝排名,
                        'Combined': 综合排名
                    })
        
        if not 结果列表:
            return None, None
        
        结果df = pd.DataFrame(结果列表)
        被淘汰者 = 结果df.loc[结果df['Combined'].idxmax(), 'Celebrity']
        
        return 被淘汰者, 结果df
    
    def 百分比法淘汰(self, season, week):
        """
        百分比法淘汰规则:
        评委百分比 + 粉丝百分比，总和最低的被淘汰
        """
        评委数据 = self.获取评委成绩(season, week)
        粉丝数据 = self.获取粉丝投票(season, week)
        
        if len(评委数据) == 0 or len(粉丝数据) == 0:
            return None, None
        
        结果列表 = []
        for _, 粉丝行 in 粉丝数据.iterrows():
            名人 = 粉丝行['Celebrity']
            评委行 = 评委数据[评委数据['Celebrity'] == 名人]
            
            if len(评委行) > 0:
                评委百分比 = 评委行.iloc[0]['Judge_Pct']
                粉丝百分比 = 粉丝行['Estimated_Fan_Pct']
                
                if pd.notna(评委百分比) and pd.notna(粉丝百分比):
                    综合百分比 = 评委百分比 + 粉丝百分比
                    结果列表.append({
                        'Celebrity': 名人,
                        'Judge_Pct': 评委百分比,
                        'Fan_Pct': 粉丝百分比,
                        'Combined': 综合百分比
                    })
        
        if not 结果列表:
            return None, None
        
        结果df = pd.DataFrame(结果列表)
        被淘汰者 = 结果df.loc[结果df['Combined'].idxmin(), 'Celebrity']
        
        return 被淘汰者, 结果df
    
    def 对比两种方法_按赛季(self, season):
        """对指定赛季的所有周次应用两种方法进行对比"""
        周次列表 = self.周度成绩[self.周度成绩['Season'] == season]['Week'].unique()
        周次列表 = sorted([w for w in 周次列表 if pd.notna(w)])
        
        对比结果 = []
        
        for week in 周次列表:
            # 获取实际淘汰者
            实际淘汰行 = self.淘汰记录[
                (self.淘汰记录['season'] == season) & 
                (self.淘汰记录['Elim_Week'] == week)
            ]
            
            if len(实际淘汰行) == 0:
                continue
            
            实际淘汰者 = 实际淘汰行.iloc[0]['celebrity_name']
            
            # 使用排名法
            排名法淘汰者, 排名法详情 = self.排名法淘汰(season, week)
            排名法正确 = (排名法淘汰者 == 实际淘汰者) if 排名法淘汰者 else False
            
            # 使用百分比法
            百分比法淘汰者, 百分比法详情 = self.百分比法淘汰(season, week)
            百分比法正确 = (百分比法淘汰者 == 实际淘汰者) if 百分比法淘汰者 else False
            
            # 检查两种方法是否一致
            方法一致 = (排名法淘汰者 == 百分比法淘汰者)
            
            对比结果.append({
                'Season': season,
                'Week': week,
                'Actual_Eliminated': 实际淘汰者,
                'Rank_Eliminated': 排名法淘汰者,
                'Pct_Eliminated': 百分比法淘汰者,
                'Rank_Correct': 排名法正确,
                'Pct_Correct': 百分比法正确,
                'Methods_Agree': 方法一致,
                'Rank_Detail': 排名法详情,
                'Pct_Detail': 百分比法详情
            })
        
        return 对比结果
    
    def 分析争议选手(self):
        """
        分析那些评委和粉丝意见分歧较大的选手
        争议标准：评委排名和粉丝排名的差距大
        """
        # 合并评委和粉丝数据
        合并数据 = self.周度成绩.merge(
            self.估计粉丝投票,
            on=['Season', 'Week', 'Celebrity'],
            how='inner'
        )
        
        # 计算分歧度量
        合并数据['评委排名_清洁'] = 合并数据['Judge_Rank'].fillna(999)
        合并数据['粉丝排名_清洁'] = 合并数据['Estimated_Fan_Rank'].fillna(999)
        合并数据['分歧度'] = np.abs(
            合并数据['评委排名_清洁'] - 合并数据['粉丝排名_清洁']
        )
        
        # 找出分歧度高的周次
        合并数据['高分歧'] = 合并数据['分歧度'] > 合并数据['分歧度'].quantile(0.75)
        
        争议数据 = 合并数据[合并数据['高分歧']].copy()
        
        # 识别最具争议的选手
        争议评分 = 争议数据.groupby(['Season', 'Celebrity']).agg({
            '分歧度': ['mean', 'count', 'max'],
            '评委排名_清洁': 'mean',
            '粉丝排名_清洁': 'mean'
        }).reset_index()
        
        争议评分.columns = ['Season', 'Celebrity', '平均分歧度', 
                                     '争议周数', '最大分歧度',
                                     '平均评委排名', '平均粉丝排名']
        
        争议评分 = 争议评分[争议评分['争议周数'] >= 3]
        争议评分 = 争议评分.sort_values('最大分歧度', ascending=False)
        
        return 争议评分, 争议数据
    
    def 分析评委否决机制(self, season, week):
        """
        模拟机制: "如果底部2名的综合得分相近，由评委最终选择谁被淘汰"
        这模拟了给评委最终否决权的情况
        返回字典包含：
        - Bottom_2: 底部2名的信息
        - Judge_Choice: 评委选择谁被淘汰
        - Percentage_Choice: 百分比法选择谁被淘汰
        """
        评委数据 = self.获取评委成绩(season, week)
        粉丝数据 = self.获取粉丝投票(season, week)
        
        if len(评委数据) == 0 or len(粉丝数据) == 0:
            return None
        
        # 使用百分比法进行组合
        结果列表 = []
        for _, 粉丝行 in 粉丝数据.iterrows():
            名人 = 粉丝行['Celebrity']
            评委行 = 评委数据[评委数据['Celebrity'] == 名人]
            
            if len(评委行) > 0:
                评委百分比 = 评委行.iloc[0]['Judge_Pct']
                粉丝百分比 = 粉丝行['Estimated_Fan_Pct']
                
                if pd.notna(评委百分比) and pd.notna(粉丝百分比):
                    综合得分 = 评委百分比 + 粉丝百分比
                    评委排名 = 评委行.iloc[0]['Judge_Rank']
                    
                    结果列表.append({
                        'Celebrity': 名人,
                        'Judge_Pct': 评委百分比,
                        'Fan_Pct': 粉丝百分比,
                        'Combined': 综合得分,
                        'Judge_Rank': 评委排名
                    })
        
        if not 结果列表:
            return None
            
        结果df = pd.DataFrame(结果列表)
        
        # 获取底部2名
        底部2名 = 结果df.nsmallest(2, 'Combined')
        
        if len(底部2名) < 2:
            return None
        
        # 百分比法的选择：综合得分最低的
        百分比法选择 = 结果df.loc[结果df['Combined'].idxmin(), 'Celebrity']
        
        # 评委选择排名最低的那个（即表现最差的）
        评委选择 = 底部2名.loc[底部2名['Judge_Rank'].idxmax(), 'Celebrity']
        
        return {
            'Bottom_2': 底部2名[['Celebrity', 'Combined']].to_dict('records'),
            'Judge_Choice': 评委选择,
            'Percentage_Choice': 百分比法选择  # 修复：改为 Percentage_Choice
        }
    
    def 生成报告(self):
        """生成完整的对比分析报告"""
        
        print("=" * 120)
        print("第二题：方法对比 - 排名法 vs 百分比法")
        print("=" * 120)
        
        # ===================================================================
        # 第一部分：总体方法准确率对比
        # ===================================================================
        
        print("\n" + "=" * 120)
        print("第一部分：两种方法的淘汰预测准确率对比（按赛季）")
        print("=" * 120)
        
        所有对比 = []
        赛季总结 = []
        
        for season in sorted(self.周度成绩['Season'].unique()):
            赛季对比结果 = self.对比两种方法_按赛季(season)
            所有对比.extend(赛季对比结果)
            
            if 赛季对比结果:
                排名法准确率 = sum([r['Rank_Correct'] for r in 赛季对比结果]) / len(赛季对比结果)
                百分比法准确率 = sum([r['Pct_Correct'] for r in 赛季对比结果]) / len(赛季对比结果)
                方法一致率 = sum([r['Methods_Agree'] for r in 赛季对比结果]) / len(赛季对比结果)
                
                赛季总结.append({
                    'Season': season,
                    'Rank_Accuracy': 排名法准确率,
                    'Pct_Accuracy': 百分比法准确率,
                    'Agreement_Rate': 方法一致率,
                    'Num_Weeks': len(赛季对比结果)
                })
                
                实际规则 = "排名法（实际使用）" if season <= 2 else "百分比法（实际使用）"
                print(f"\n✓ 第 {season} 季（{实际规则}）:")
                print(f"  • 排名法准确率：{排名法准确率*100:.1f}%")
                print(f"  • 百分比法准确率：{百分比法准确率*100:.1f}%")
                print(f"  • 两种方法一致率：{方法一致率*100:.1f}%")
                print(f"  • 分析周次总数：{len(赛季对比结果)}")
                
                if 排名法准确率 > 百分比法准确率:
                    优势 = (排名法准确率 - 百分比法准确率) * 100
                    print(f"  ➜ 排名法更准确，领先 {优势:.1f}%")
                elif 百分比法准确率 > 排名法准确率:
                    优势 = (百分比法准确率 - 排名法准确率) * 100
                    print(f"  ➜ 百分比法更准确，领先 {优势:.1f}%")
                else:
                    print(f"  ➜ 两种方法准确率相同")
        
        赛季总结df = pd.DataFrame(赛季总结)
        
        # 总体统计
        print("\n" + "-" * 120)
        print("【总体统计】")
        print(f"✓ 排名法平均准确率：{赛季总结df['Rank_Accuracy'].mean()*100:.1f}%")
        print(f"✓ 百分比法平均准确率：{赛季总结df['Pct_Accuracy'].mean()*100:.1f}%")
        print(f"✓ 两种方法平均一致率：{赛季总结df['Agreement_Rate'].mean()*100:.1f}%")
        
        # 哪种方法更受欢迎
        排名法胜利数 = (赛季总结df['Rank_Accuracy'] > 赛季总结df['Pct_Accuracy']).sum()
        百分比法胜利数 = (赛季总结df['Pct_Accuracy'] > 赛季总结df['Rank_Accuracy']).sum()
        
        print(f"\n✓ 方法对比（按赛季胜负）:")
        print(f"  • 排名法在 {排名法胜利数} 个赛季表现更好")
        print(f"  • 百分比法在 {百分比法胜利数} 个赛季表现更好")
        
        if 百分比法胜利数 > 排名法胜利数:
            print(f"  ➜ 【结论】百分比法更符合观众意见")
        else:
            print(f"  ➜ 【结论】排名法更符合观众意见")
        
        # ===================================================================
        # 第二部分：争议选手分析
        # ===================================================================
        
        print("\n" + "=" * 120)
        print("第二部分：争议选手分析 - 评委与观众意见不一致的情况")
        print("=" * 120)
        
        争议评分, 争议数据 = self.分析争议选手()
        
        print(f"\n✓ 十大最具争议的选手（评委与粉丝意见分歧最大）:")
        print("-" * 120)
        
        for 排名, (_, 行数据) in enumerate(争议评分.head(10).iterrows(), 1):
            print(f"\n{排名}. 第 {int(行数据['Season'])} 季 - {行数据['Celebrity']}")
            print(f"   • 最大分歧度：{行数据['最大分歧度']:.1f} 个排名")
            print(f"   • 平均分歧度：{行数据['平均分歧度']:.1f} 个排名")
            print(f"   • 争议周数：{int(行数据['争议周数'])} 周")
            print(f"   • 平均评委排名：{行数据['平均评委排名']:.1f}")
            print(f"   • 平均粉丝排名：{行数据['平均粉丝排名']:.1f}")
            
            if 行数据['平均评委排名'] < 行数据['平均粉丝排名']:
                print(f"   ➜ 现象分析：评委偏爱这位选手")
                print(f"      评委排名 {行数据['平均评委排名']:.1f} vs 粉丝排名 {行数据['平均粉丝排名']:.1f}")
            else:
                print(f"   ➜ 现象分析：粉丝偏爱这位选手")
                print(f"      粉丝排名 {行数据['平均粉丝排名']:.1f} vs 评委排名 {行数据['平均评委排名']:.1f}")
        
        # 分析特殊的争议案例
        print("\n" + "-" * 120)
        print("【特殊争议案例分析】")
        print("-" * 120)
        
        案例研究列表 = [
            (2, "Jerry Rice", "多次获得低评委分，但最终获得亚军"),
            (4, "Billy Ray Cyrus", "6周获得最低评委分，但最终获得第5名"),
        ]
        
        for season, 名人, 描述 in 案例研究列表:
            案例数据 = 争议数据[
                (争议数据['Season'] == season) & 
                (争议数据['Celebrity'] == 名人)
            ]
            
            if len(案例数据) > 0:
                print(f"\n✓ 第 {season} 季 - {名人}：{描述}")
                print(f"  • 总出场次数：{len(案例数据)} 周")
                print(f"  • 平均评委排名：{案例数据['评委排名_清洁'].mean():.1f}")
                print(f"  • 平均粉丝排名：{案例数据['粉丝排名_清洁'].mean():.1f}")
                print(f"  • 平均分歧度：{案例数据['分歧度'].mean():.1f}")
                
                # 获取最终成绩
                选手数据 = self.选手特征[
                    (self.选手特征['Name'] == 名人) & 
                    (self.选手特征['Season'] == season)
                ]
                if len(选手数据) > 0:
                    最终名次 = 选手数据.iloc[0]['Final_Placement']
                    print(f"  • 最终排名：第 {int(最终名次)} 名")
                    print(f"  ➜ 这说明：评委的低分并未能阻止他晋级，观众投票权重起了关键作用")
        
        # ===================================================================
        # 第三部分：方法不一致性分析
        # ===================================================================
        
        print("\n" + "=" * 120)
        print("第三部分：两种方法的结果差异分析")
        print("=" * 120)
        
        所有对比df = pd.DataFrame(所有对比)
        
        不一致案例 = 所有对比df[~所有对比df['Methods_Agree']].copy()
        print(f"\n✓ 方法不一致的淘汰决策总数：{len(不一致案例)} 例（总共 {len(所有对比df)} 周）")
        print(f"  方法不一致率：{len(不一致案例)/len(所有对比df)*100:.1f}%")
        
        if len(不一致案例) > 0:
            print(f"\n【当两种方法结果不同时的分析】:")
            
            排名法在不一致中正确数 = (不一致案例['Rank_Correct']).sum()
            百分比法在不一致中正确数 = (不一致案例['Pct_Correct']).sum()
            两个都错数 = len(不一致案例) - 排名法在不一致中正确数 - 百分比法在不一致中正确数
            
            print(f"  • 排名法做出了正确预测：{排名法在不一致中正确数} 例")
            print(f"  • 百分比法做出了正确预测：{百分比法在不一致中正确数} 例")
            print(f"  • 两种方法都预测错误：{两个都错数} 例")
            
            if 百分比法在不一致中正确数 > 排名法在不一致中正确数:
                差距 = 百分比法在不一致中正确数 - 排名法在不一致中正确数
                print(f"\n  ➜ 【关键发现】当两种方法出现分歧时，百分比法更准确！")
                print(f"     百分比法多预测对 {差距} 例不一致的情况")
            elif 排名法在不一致中正确数 > 百分比法在不一致中正确数:
                差距 = 排名法在不一致中正确数 - 百分比法在不一致中正确数
                print(f"\n  ➜ 【关键发现】当两种方法出现分歧时，排名法更准确！")
                print(f"     排名法多预测对 {差距} 例不一致的情况")
            else:
                print(f"\n  ➜ 【关键发现】当两种方法出现分歧时，两者准确率相同")
        
        # ===================================================================
        # 第四部分：评委否决机制的影响
        # ===================================================================
        
        print("\n" + "=" * 120)
        print("第四部分：如果加入'评委在底部2名中选择'的否决机制")
        print("=" * 120)
        
        评委机制结果 = []
        
        for season in sorted(self.周度成绩['Season'].unique()):
            周次列表 = self.周度成绩[self.周度成绩['Season'] == season]['Week'].unique()
            周次列表 = sorted([w for w in 周次列表 if pd.notna(w)])
            
            for week in 周次列表:
                实际淘汰行 = self.淘汰记录[
                    (self.淘汰记录['season'] == season) & 
                    (self.淘汰记录['Elim_Week'] == week)
                ]
                
                if len(实际淘汰行) == 0:
                    continue
                
                实际淘汰者 = 实际淘汰行.iloc[0]['celebrity_name']
                
                机制结果 = self.分析评委否决机制(season, week)
                
                if 机制结果:
                    评委机制结果.append({
                        'Season': season,
                        'Week': week,
                        'Actual_Eliminated': 实际淘汰者,
                        'Judge_Choice': 机制结果['Judge_Choice'],
                        'Percentage_Choice': 机制结果['Percentage_Choice'],
                        'Judge_Correct': 机制结果['Judge_Choice'] == 实际淘汰者,
                        'Percentage_Correct': 机制结果['Percentage_Choice'] == 实际淘汰者
                    })
        
        if 评委机制结果:
            评委机制df = pd.DataFrame(评委机制结果)
            
            评委准确率 = 评委机制df['Judge_Correct'].sum() / len(评委机制df) if len(评委机制df) > 0 else 0
            百分比准确率 = 评委机制df['Percentage_Correct'].sum() / len(评委机制df) if len(评委机制df) > 0 else 0
            
            print(f"\n✓ 评委否决机制的效果对比:")
            print(f"  • 纯百分比法准确率：{百分比准确率*100:.1f}%")
            print(f"  • 加入评委否决后准确率：{评委准确率*100:.1f}%")
            print(f"  • 分析样本数：{len(评委机制df)} 周")
            
            if 评委准确率 > 百分比准确率:
                改进 = (评委准确率 - 百分比准确率) * 100
                print(f"  ➜ 评委否决机制改进了准确率 {改进:.1f}%")
                print(f"  ➜ 【建议】保留评委否决机制")
            elif 百分比准确率 > 评委准确率:
                损害 = (百分比准确率 - 评委准确率) * 100
                print(f"  ➜ 评委否决机制降低了准确率 {损害:.1f}%")
                print(f"  ➜ 【建议】去除评委否决机制")
            else:
                print(f"  ➜ 评委否决机制对准确率没有影响")
        else:
            print(f"\n⚠ 没有足够的数据来评估评委否决机制的影响")
        
        # ===================================================================
        # 第五部分：最终建议
        # ===================================================================
        
        print("\n" + "=" * 120)
        print("第五部分：基于分析的推荐方案")
        print("=" * 120)
        
        print(f"\n【推荐使用的方法】")
        print("-" * 120)
        
        if len(赛季总结df) > 0:
            if 赛季总结df['Pct_Accuracy'].mean() > 赛季总结df['Rank_Accuracy'].mean():
                优势 = (赛季总结df['Pct_Accuracy'].mean() - 赛季总结df['Rank_Accuracy'].mean()) * 100
                print(f"✓ 推荐：使用 【百分比法】")
                print(f"  理由：准确率比排名法高 {优势:.1f}%")
                print(f"  理由：更符合观众意见和偏好")
                print(f"  理由：减少争议和不公平决策")
            else:
                优势 = (赛季总结df['Rank_Accuracy'].mean() - 赛季总结df['Pct_Accuracy'].mean()) * 100
                print(f"✓ 推荐：使用 【排名法】")
                print(f"  理由：准确率比百分比法高 {优势:.1f}%")
        
        print(f"\n【评委否决机制的建议】")
        print("-" * 120)
        
        if 评委机制结果:
            if 评委准确率 > 百分比准确率:
                print(f"✓ 建议：【保留】评委否决机制")
                print(f"  理由：可以改进决策准确性")
                print(f"  理由：减少不可预测的淘汰结果")
            else:
                print(f"✓ 建议：【取消】评委否决机制")
                print(f"  理由：降低了准确率")
                print(f"  理由：应该更尊重观众投票")
        else:
            print(f"⚠ 数据不足，无法做出评委否决机制的建议")
        
        print(f"\n【公平性考虑】")
        print("-" * 120)
        if len(赛季总结df) > 0:
            print(f"✓ 当前方法一致率：{赛季总结df['Agreement_Rate'].mean()*100:.1f}%")
            print(f"  （约 {赛季总结df['Agreement_Rate'].mean()*100:.0f}% 的淘汰在两种方法下结果相同）")
        
        if len(争议评分) > 0:
            print(f"\n✓ 发现的争议选手总数：{len(争议评分)} 位")
            print(f"  这些选手的平均分歧度：{争议评分['最大分歧度'].mean():.1f} 个排名")
            print(f"  这反映了评委和观众的意见有时会有很大分歧")
        
        print("\n" + "=" * 120)
        
        return {
            'season_summary': 赛季总结df,
            'comparisons': 所有对比df,
            'controversy': 争议评分,
            'judge_mechanism': 评委机制df if 评委机制结果 else None
        }

if __name__ == "__main__":
    分析器 = 方法对比分析()
    结果 = 分析器.生成报告()