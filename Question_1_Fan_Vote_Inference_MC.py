import pandas as pd
import numpy as np
import itertools

# =========================================================
# 1. 基础工具函数 (保持不变)
# =========================================================

def softmax(x):
    """将得分转换为概率分布"""
    # 减去最大值防止溢出
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def fan_rank_from_pct(pct):
    """根据得票率计算排名 (票多=Rank1)"""
    return np.argsort(np.argsort(-pct)) + 1

def eliminated_by_rule(judge, fan_pct, use_rank):
    """
    根据规则判断谁被淘汰
    返回: 被淘汰者的索引 (index)
    """
    if use_rank:
        # 排名制：JudgeRank + FanRank，总分最大者淘汰
        fan_rank = fan_rank_from_pct(fan_pct)
        total = judge + fan_rank
        # 如果有并列最大，通常看裁判分或粉丝分，这里简化为取第一个最大
        return np.argmax(total)
    else:
        # 百分比制：JudgePct + FanPct，总分最小者淘汰
        total = judge + fan_pct
        return np.argmin(total)

# =========================================================
# 2. 核心模块：参数调优 & 后验模拟
# =========================================================

def tune_parameters_for_season(season, season_weekly, elim_data, features, weight_dict):
    """
    针对特定赛季，通过网格搜索寻找最佳的 (Alpha, Beta)
    Alpha: 粉丝意愿权重 (Popularity的系数)
    Beta:  裁判影响权重 (Judge的系数)
    """
    # 搜索空间：涵盖从“完全看裁判”到“完全看粉丝”的各种情况
    alpha_grid = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0] 
    beta_grid = [0.0, 0.5, 1.0, 1.5] # Beta=0 表示粉丝投票完全无视裁判分
    
    best_acc = -1
    best_params = (1.5, 0.5) # 默认值
    
    # 提取该赛季所有有淘汰发生的周
    valid_weeks = season_weekly['Week'].unique()
    
    for alpha, beta in itertools.product(alpha_grid, beta_grid):
        correct_count = 0
        total_checks = 0
        
        for week in valid_weeks:
            # 获取当周数据
            group = season_weekly[season_weekly['Week'] == week]
            elim_row = elim_data[(elim_data['season'] == season) & (elim_data['Elim_Week'] == week)]
            
            if elim_row.empty: continue
            target_name = elim_row['celebrity_name'].values[0]
            if target_name not in group['Celebrity'].values: continue
            
            # 1. 计算基础流行度分数 (Pop Score)
            pop_scores = []
            for _, row in group.iterrows():
                key = (row['Celebrity'], season)
                score = 0
                if key in features.index:
                    feat_row = features.loc[key]
                    for f, w in weight_dict.items():
                        if f in feat_row: score += w * feat_row[f]
                pop_scores.append(score)
            pop_scores = np.array(pop_scores)
            
            # 2. 获取裁判数据
            use_rank = (season <= 2) or (season >= 28)
            judge = group['Judge_Rank'].values if use_rank else group['Judge_Pct'].values
            
            # 3. 确定性预测 (不加噪声)
            # 假设粉丝倾向公式：F = alpha * Pop + beta * Judge_Goodness
            # Judge_Goodness: 在排名制里 rank越小越好(负相关)，在百分比制里 pct越大越好(正相关)
            if use_rank:
                # Rank制：裁判Rank越小越好。我们假设粉丝喜欢裁判分高的人? 
                # 这里为了统一：我们构建 "Latent Score" (越大越好)
                # Pop越大越好。JudgeRank越小越好 -> -JudgeRank
                F = alpha * pop_scores + beta * judge 
            else:
                # Percent制：JudgePct越大越好
                F = alpha * pop_scores - beta * judge
            
            pred_pct = softmax(F)
            
            # 4. 检查是否命中真实淘汰者
            target_idx = group[group['Celebrity'] == target_name].index[0] - group.index[0]
            pred_elim_idx = eliminated_by_rule(judge, pred_pct, use_rank)
            
            if pred_elim_idx == target_idx:
                correct_count += 1
            total_checks += 1
            
        if total_checks > 0:
            acc = correct_count / total_checks
            if acc > best_acc:
                best_acc = acc
                best_params = (alpha, beta)
                
    return best_params, best_acc

# =========================================================
# 3. 主程序：蒙特卡洛反演
# =========================================================

def infer_fan_votes_final():
    print("Step 1: Loading and Preprocessing Data...")
    
    weekly = pd.read_csv("Weekly_Performance.csv")
    elim = pd.read_csv("Elimination_Lookup.csv")
    features = pd.read_csv("Contestant_Features.csv")
    
    # -----------------------------------------------------
    # 改进点 1: 特征标准化 (Feature Scaling)
    # 必须做！否则 Age(50) 会比 Partner_Strength(8) 权重高太多
    # -----------------------------------------------------
    cols_to_norm = ['Age', 'Partner_Strength_Index']
    for col in cols_to_norm:
        if col in features.columns:
            mean_val = features[col].mean()
            std_val = features[col].std()
            features[col] = (features[col] - mean_val) / (std_val + 1e-6)
            
    # 设置索引方便查找
    if 'Season' in features.columns and 'Name' in features.columns:
        features = features.set_index(['Name', 'Season'])
    
    # 特征权重 (来自你的 RF 结果)
    # 假设 Partner_Strength_Index 越高代表舞伴越强，Age 越小越好? 
    # RF 只能给绝对值。这里假设：
    # Age: 负相关 (越年轻越好) -> 权重应为负? 或者在特征里取反?
    # 这里我们假设 RF 图表中的权重是"影响力绝对值"。
    # 修正逻辑：更年轻(Age小) -> 分高; 舞伴强(Index高) -> 分高
    # 由于我们做了 Z-Score，Age小的 Z-Score 是负数。
    # 所以 Age 的权重系数应该是 *负数*，才能让年轻人的 Score 变高。
    weight_dict = {
        'Age': -0.45,                 # 负号：年龄越小，得分越高
        'Partner_Strength_Index': 0.40, # 正号：舞伴越强，得分越高
        'Region_West': 0.05,
        'Region_South': 0.05
    }

    results = []
    
    # 按赛季循环
    for season, season_group in weekly.groupby('Season'):
        print(f"Processing Season {season}...")
        
        # -------------------------------------------------
        # 改进点 2: 赛季参数自适应 (Adaptive Parameter Tuning)
        # -------------------------------------------------
        (best_alpha, best_beta), acc = tune_parameters_for_season(
            season, season_group, elim, features, weight_dict
        )
        print(f"  -> Best Params: Alpha={best_alpha}, Beta={best_beta} (Hist Acc: {acc:.2%})")
        
        for week, group in season_group.groupby('Week'):
            # 查找真实淘汰者
            elim_row = elim[(elim['season'] == season) & (elim['Elim_Week'] == week)]
            if elim_row.empty: continue
            
            target_star = elim_row['celebrity_name'].values[0]
            if target_star not in group['Celebrity'].values: continue
            
            # 准备基础数据
            pop_scores = []
            for _, row in group.iterrows():
                key = (row['Celebrity'], season)
                score = 0
                if key in features.index:
                    feat_row = features.loc[key]
                    for f, w in weight_dict.items():
                        if f in feat_row: score += w * feat_row[f]
                pop_scores.append(score)
            pop_scores = np.array(pop_scores)
            
            use_rank = (season <= 2) or (season >= 28)
            judge = group['Judge_Rank'].values if use_rank else group['Judge_Pct'].values
            target_idx = group[group['Celebrity'] == target_star].index[0] - group.index[0]
            
            # -------------------------------------------------
            # 改进点 3: 后验过滤 (Rejection Sampling)
            # -------------------------------------------------
            valid_samples = []
            max_attempts = 20000  # 最大尝试次数
            required_samples = 200 # 我们需要多少个有效样本来算方差
            
            sigma = 0.3 # 噪声水平
            
            for _ in range(max_attempts):
                if len(valid_samples) >= required_samples:
                    break
                
                # 添加随机噪声 (Noise Injection)
                noise = np.random.normal(0, sigma, len(group))
                
                # 计算潜在得分 Latent Score
                if use_rank:
                    # Rank制：JudgeRank越小越好。我们希望 Score 越大越好。
                    # 所以减去 beta * judge (因为 judge 大是坏事)
                    F = best_alpha * pop_scores + best_beta * judge + noise
                else:
                    # Percent制：JudgePct 越大越好。
                    F = best_alpha * pop_scores - best_beta * judge + noise
                
                # 转换为百分比
                fan_pct = softmax(F)
                
                # 检查：这个随机投票，是否会导致 target_star 被淘汰？
                if eliminated_by_rule(judge, fan_pct, use_rank) == target_idx:
                    valid_samples.append(fan_pct)
            
            # -------------------------------------------------
            # 改进点 4: 统计与不确定性 (Certainty Measure)
            # -------------------------------------------------
            if len(valid_samples) > 10:
                # 成功复现：计算有效样本的均值和标准差
                valid_samples = np.array(valid_samples)
                mean_votes = valid_samples.mean(axis=0)
                std_votes = valid_samples.std(axis=0) # 这就是 Certainty_Std
                note = "Converged"
            else:
                # 失败复现 (黑天鹅事件)：模型无法解释这次淘汰
                # Fallback: 返回一个基于最佳参数的无噪声估计，但给予极高的不确定性
                if use_rank:
                    F_base = best_alpha * pop_scores + best_beta * judge
                else:
                    F_base = best_alpha * pop_scores - best_beta * judge
                mean_votes = softmax(F_base)
                std_votes = np.full(len(group), 0.15) # 0.15 是很大的标准差
                note = "Failed_to_Fit"

            # 保存结果
            fan_rank_est = fan_rank_from_pct(mean_votes)
            
            for i, (_, row) in enumerate(group.iterrows()):
                results.append({
                    'Season': season,
                    'Week': week,
                    'Celebrity': row['Celebrity'],
                    'Judge_Score': row['Judge_Rank'] if use_rank else row['Judge_Pct'],
                    'Estimated_Fan_Pct': mean_votes[i],
                    'Estimated_Fan_Rank': fan_rank_est[i],
                    'Certainty_Std': std_votes[i], # 核心指标
                    'Model_Alpha': best_alpha,
                    'Model_Beta': best_beta,
                    'Fit_Status': note
                })

    # 输出文件
    df_out = pd.DataFrame(results)
    output_path = "Estimated_Fan_Votes_Final_Model_2.csv"
    df_out.to_csv(output_path, index=False)
    print(f"All done! Results saved to {output_path}")
    print(f"Total rows: {len(df_out)}")

if __name__ == "__main__":
    infer_fan_votes_final()