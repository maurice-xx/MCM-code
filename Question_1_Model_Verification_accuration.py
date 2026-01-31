import pandas as pd
import numpy as np
from collections import defaultdict

def validate_fan_votes():
    """验证 p3 推出的粉丝投票估计值"""
    
    # 1. 加载数据
    try:
        estimated = pd.read_csv("Estimated_Fan_Votes_Final_Model_2.csv")
        weekly = pd.read_csv("Weekly_Performance.csv")
        elim = pd.read_csv("Elimination_Lookup.csv")
    except FileNotFoundError as e:
        print(f"❌ 数据文件缺失: {e}")
        return
    
    print("=" * 80)
    print("粉丝投票估计值验证 - 区分两种淘汰机制")
    print("=" * 80)
    
    # 2. 合并数据：加入规则信息
    # 根据赛季判断规则（假设第1,2赛季是Rank制，第3+赛季是Percent制）
    def get_rule(season):
        return "Rank" if season <= 2 else "Percent"
    
    estimated['Rule'] = estimated['Season'].apply(get_rule)
    weekly['Rule'] = weekly['Season'].apply(get_rule)
    
    # 3. 验证 Rank 制 (Season 1-2)
    print("\n【Rank 制验证】(Season 1-2)")
    print("-" * 80)
    rank_data = estimated[estimated['Rule'] == 'Rank']
    
    if len(rank_data) > 0:
        # 3a. 检查 Estimated_Fan_Rank 是否为有效排名
        rank_by_week = rank_data.groupby(['Season', 'Week'])
        valid_rank_count = 0
        total_weeks = 0
        
        for (season, week), group in rank_by_week:
            total_weeks += 1
            n_contestants = len(group)
            ranks = sorted(group['Estimated_Fan_Rank'].values)
            expected_ranks = list(range(1, n_contestants + 1))
            
            if ranks == expected_ranks:
                valid_rank_count += 1
            else:
                print(f"  ⚠️ S{season}W{week}: 排名不是 1-{n_contestants} 的排列")
                print(f"     期望: {expected_ranks}, 实际: {ranks}")
        
        print(f"  ✅ 有效排名周数: {valid_rank_count}/{total_weeks}")
        
        # 3b. 验证淘汰逻辑 (Judge_Rank + Fan_Rank 最大者被淘汰)
        print("\n  淘汰逻辑验证 (Judge_Rank + Fan_Rank 最大→淘汰):")
        correct_elim = 0
        total_elim = 0
        
        for (season, week), group in rank_by_week:
            # 合并 Judge_Rank
            judge_data = weekly[(weekly['Season'] == season) & (weekly['Week'] == week)]
            
            for idx, row in group.iterrows():
                celebrity = row['Celebrity']
                judge_row = judge_data[judge_data['Celebrity'] == celebrity]
                
                if len(judge_row) > 0:
                    judge_rank = judge_row.iloc[0]['Judge_Rank']
                    fan_rank = row['Estimated_Fan_Rank']
                    combined_rank = judge_rank + fan_rank
                    
                    # 查看该周是否有人被淘汰
                    elim_row = elim[(elim['season'] == season) & 
                                    (elim['Elim_Week'] == week)]
                    
                    if len(elim_row) > 0:
                        elim_celebrity = elim_row.iloc[0]['celebrity_name']
                        total_elim += 1
                        
                        # 找出该周谁的 combined_rank 最大
                        week_combined = []
                        for _, r in group.iterrows():
                            c = r['Celebrity']
                            j_row = judge_data[judge_data['Celebrity'] == c]
                            if len(j_row) > 0:
                                j_rank = j_row.iloc[0]['Judge_Rank']
                                f_rank = r['Estimated_Fan_Rank']
                                week_combined.append((c, j_rank + f_rank))
                        
                        if week_combined:
                            max_celebrity = max(week_combined, key=lambda x: x[1])[0]
                            if max_celebrity == elim_celebrity:
                                correct_elim += 1
        
        if total_elim > 0:
            print(f"  ✅ Rank制淘汰预测准确率: {correct_elim}/{total_elim} = {100*correct_elim/total_elim:.1f}%")
    else:
        print("  (无 Rank 制数据)")
    
    # 4. 验证 Percent 制 (Season 3+)
    print("\n【Percent 制验证】(Season 3+)")
    print("-" * 80)
    percent_data = estimated[estimated['Rule'] == 'Percent']
    
    if len(percent_data) > 0:
        # 4a. 检查 Estimated_Fan_Pct 求和是否为 1.0
        percent_by_week = percent_data.groupby(['Season', 'Week'])
        sum_to_one_count = 0
        total_weeks_percent = 0
        
        for (season, week), group in percent_by_week:
            total_weeks_percent += 1
            pct_sum = group['Estimated_Fan_Pct'].sum()
            
            if abs(pct_sum - 1.0) < 0.01:  # 允许 1% 误差
                sum_to_one_count += 1
            else:
                print(f"  ⚠️ S{season}W{week}: 粉丝百分比求和 = {pct_sum:.4f} (期望 1.0)")
        
        print(f"  ✅ 百分比求和有效周数: {sum_to_one_count}/{total_weeks_percent}")
        
        # 4b. 验证淘汰逻辑 (Judge_Pct + Fan_Pct 最小→淘汰)
        print("\n  淘汰逻辑验证 (Judge_Pct + Fan_Pct 最小→淘汰):")
        correct_elim_percent = 0
        total_elim_percent = 0
        
        for (season, week), group in percent_by_week:
            # 合并 Judge_Pct
            judge_data = weekly[(weekly['Season'] == season) & (weekly['Week'] == week)]
            
            elim_row = elim[(elim['season'] == season) & 
                           (elim['Elim_Week'] == week)]
            
            if len(elim_row) > 0:
                elim_celebrity = elim_row.iloc[0]['celebrity_name']
                total_elim_percent += 1
                
                # 找出该周谁的 combined_pct 最小
                week_combined_pct = []
                for _, r in group.iterrows():
                    c = r['Celebrity']
                    j_row = judge_data[judge_data['Celebrity'] == c]
                    if len(j_row) > 0:
                        j_pct = j_row.iloc[0]['Judge_Pct']
                        f_pct = r['Estimated_Fan_Pct']
                        week_combined_pct.append((c, j_pct + f_pct))
                
                if week_combined_pct:
                    min_celebrity = min(week_combined_pct, key=lambda x: x[1])[0]
                    if min_celebrity == elim_celebrity:
                        correct_elim_percent += 1
        
        if total_elim_percent > 0:
            print(f"  ✅ Percent制淘汰预测准确率: {correct_elim_percent}/{total_elim_percent} = {100*correct_elim_percent/total_elim_percent:.1f}%")
    else:
        print("  (无 Percent 制数据)")
    
    # 5. 总体统计
    print("\n【总体统计】")
    print("-" * 80)
    print(f"  总行数: {len(estimated)}")
    print(f"  Rank制周数: {len(rank_data.groupby(['Season', 'Week']))}")
    print(f"  Percent制周数: {len(percent_data.groupby(['Season', 'Week']))}")
    print(f"  粉丝投票标准差范围: [{estimated['Certainty_Std'].min():.4f}, {estimated['Certainty_Std'].max():.4f}]")
    print("\n✅ 验证完成！")

if __name__ == "__main__":
    validate_fan_votes()