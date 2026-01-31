import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

"""
分析脚本：模拟“混合制（方案3：叙事加权 + 方案4：透明排名）”
在争议赛季下的实现效果，并生成可视化图。

输出：
 - 图像文件：'MixedPlan_ControversialSeasons_overview.png' 等
 - CSV文件：'MixedPlan_ControversialSeasons_summary.csv'

用法：直接运行脚本或在交互式环境中调用 main()
"""

sns.set(style='whitegrid', font='sans-serif', font_scale=1.05)

# --- 核心函数 ----------------------------------------------------

def load_data():
    weekly = pd.read_csv('Weekly_Performance.csv')
    estimated = pd.read_csv('Estimated_Fan_Votes_Final_Model_2.csv')
    merged = weekly.merge(
        estimated[['Season', 'Week', 'Celebrity', 'Estimated_Fan_Pct']],
        on=['Season', 'Week', 'Celebrity'], how='inner'
    )
    merged['Judge_Fan_Gap'] = (merged['Judge_Pct'] - merged['Estimated_Fan_Pct']).abs()
    return merged


def compute_plan3_storytelling(df):
    # 计算Week1基线
    week1 = df[df['Week'] == df.groupby(['Season', 'Celebrity'])['Week'].transform('min')]
    week1_scores = week1[['Season', 'Celebrity', 'Judge_Pct']].rename(columns={'Judge_Pct': 'Week1_Judge_Pct'})

    df = df.merge(week1_scores, on=['Season', 'Celebrity'], how='left')
    df['Improvement_Ratio'] = ((df['Judge_Pct'] - df['Week1_Judge_Pct']) / (df['Week1_Judge_Pct'] + 1e-6)).clip(-1, 2)
    df['Improvement_Bonus'] = np.maximum(df['Improvement_Ratio'], 0)

    # 稳定性（以赛季-选手为单位计算std）
    consistency = df.groupby(['Season', 'Celebrity'])['Judge_Pct'].std()
    consistency_map = consistency.to_dict()
    df['Consistency_Score'] = df.apply(
        lambda row: 1 - (consistency_map.get((row['Season'], row['Celebrity']), 0) / 10), axis=1
    ).clip(0, 1)

    # 故事潜力（用年龄和改进综合）
    if 'celebrity_age_during_season' in df.columns:
        df['Age'] = df['celebrity_age_during_season']
        age_min, age_max = df['Age'].min(), df['Age'].max()
        df['Story_Potential'] = (
            (df['Age'] - age_min) / (age_max - age_min + 1e-6) * 0.5 +
            df['Improvement_Ratio'].clip(0, 1) * 0.5
        ).clip(0, 1)
    else:
        df['Story_Potential'] = df['Improvement_Ratio'].clip(0, 1)

    df['Combined_Score_Plan3_raw'] = (
        df['Judge_Pct'] * 0.40 +
        df['Estimated_Fan_Pct'] * 0.30 +
        df['Improvement_Bonus'] * 0.15 +
        df['Story_Potential'] * 0.10 +
        df['Consistency_Score'] * 0.05
    )

    # 标准化到0-1，方便比较
    scaler = MinMaxScaler()
    df['Combined_Score_Plan3'] = scaler.fit_transform(df[['Combined_Score_Plan3_raw']]).flatten()

    return df


def compute_rankings(df):
    # 现有 50-50 制度
    df['Existing_Combined'] = 0.5 * df['Judge_Pct'] + 0.5 * df['Estimated_Fan_Pct']
    df['Existing_Rank'] = df.groupby(['Season', 'Week'])['Existing_Combined'].rank(ascending=False, method='min')

    # 评委/观众排名
    df['Judge_Rank'] = df.groupby(['Season', 'Week'])['Judge_Pct'].rank(ascending=False, method='min')
    df['Fan_Rank'] = df.groupby(['Season', 'Week'])['Estimated_Fan_Pct'].rank(ascending=False, method='min')

    # 混合制：使用 Plan3 的得分，并按周进行排名（更接近“最终排名”）
    df['Mixed_Score'] = df['Combined_Score_Plan3']
    df['Mixed_Rank'] = df.groupby(['Season', 'Week'])['Mixed_Score'].rank(ascending=False, method='min')

    # 排名差异
    df['RankDiff_Judge_Fan'] = (df['Judge_Rank'] - df['Fan_Rank']).abs()
    df['RankDiff_Existing_Mixed'] = (df['Existing_Rank'] - df['Mixed_Rank']).abs()
    df['RankDiff_Judge_Mixed'] = (df['Judge_Rank'] - df['Mixed_Rank']).abs()

    return df


def identify_controversial_seasons(df, top_n=3, gap_threshold=None):
    season_gap = df.groupby('Season')['Judge_Fan_Gap'].mean().sort_values(ascending=False)
    if gap_threshold is not None:
        selected = season_gap[season_gap > gap_threshold].index.tolist()
    else:
        selected = season_gap.head(top_n).index.tolist()
    return selected, season_gap


# --- 可视化 ----------------------------------------------------

def plot_overview_for_seasons(df, seasons, out_png='MixedPlan_ControversialSeasons_overview.png'):
    figs = []
    plt.figure(figsize=(14, 10))

    # 子图1：每节赛季 - 平均排名差异随周变化（评委 vs 观众 vs 混合）
    plt.subplot(2, 1, 1)
    for s in seasons:
        temp = df[df['Season'] == s].groupby('Week').agg({
            'RankDiff_Judge_Fan': 'mean',
            'RankDiff_Existing_Mixed': 'mean'
        }).reset_index()
        plt.plot(temp['Week'], temp['RankDiff_Judge_Fan'], marker='o', label=f'S{int(s)} Judge vs Fan')
        plt.plot(temp['Week'], temp['RankDiff_Existing_Mixed'], marker='x', linestyle='--', label=f'S{int(s)} Existing vs Mixed')
    plt.xlabel('Week')
    plt.ylabel('Avg Rank Difference')
    plt.title('Weekly Avg Rank Difference (Judge vs Fan) and (Existing 50-50 vs Mixed)')
    plt.legend()

    # 子图2：每赛季淘汰差异计数（混合制 vs 现有）
    plt.subplot(2, 1, 2)
    elim_change = []
    seasons_sorted = sorted(seasons)
    for s in seasons_sorted:
        s_df = df[df['Season'] == s]
        # 按周比较被淘汰者（最低分）
        changes = 0
        total = 0
        weeks = s_df['Week'].unique()
        for w in weeks:
            week_df = s_df[s_df['Week'] == w]
            if len(week_df) < 2:
                continue
            lowest_existing = week_df.loc[week_df['Existing_Combined'].idxmin(), 'Celebrity']
            lowest_mixed = week_df.loc[week_df['Mixed_Score'].idxmin(), 'Celebrity']
            if lowest_existing != lowest_mixed:
                changes += 1
            total += 1
        elim_change.append({'Season': s, 'WeeksCompared': total, 'ElimDiffCount': changes})
    elim_df = pd.DataFrame(elim_change)
    sns.barplot(data=elim_df, x='Season', y='ElimDiffCount', palette='viridis')
    plt.ylabel('Number of Weeks with Different Elimination')
    plt.title('Per-season: Number of Weeks with Different Eliminations (Mixed vs Existing 50-50)')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✓ Saved overview plot: {out_png}")


def plot_detailed_season(df, season, out_png_prefix='MixedPlan_Season'):
    s_df = df[df['Season'] == season].copy()
    if s_df.empty:
        print(f"⚠️ No data found for season {season}")
        return

    # 排名前 5 的周展示（示例：周 x 的排名热力图）
    pivot = s_df.pivot_table(index='Celebrity', columns='Week', values='RankDiff_Judge_Fan', aggfunc='mean')
    plt.figure(figsize=(12, max(6, len(pivot) * 0.25)))
    sns.heatmap(pivot.fillna(0), cmap='coolwarm', center=0)
    plt.title(f'S{int(season)} - Weekly Rank Difference: Judge vs Fan (heatmap)')
    plt.xlabel('Week')
    plt.ylabel('Celebrity')
    png = f"{out_png_prefix}_{int(season)}_rankdiff_heatmap.png"
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.close()
    print(f"✓ Saved heatmap: {png}")

    # 散点图：评委分 vs 观众分，点大小表示混合得分
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=s_df, x='Judge_Pct', y='Estimated_Fan_Pct', size='Mixed_Score', hue='Mixed_Score', palette='Spectral', sizes=(20, 200), alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.title(f'S{int(season)} - Judge vs Fan (point size indicates Mixed Score)')
    plt.xlabel('Judge_Pct')
    plt.ylabel('Estimated_Fan_Pct')
    png2 = f"{out_png_prefix}_{int(season)}_judge_vs_fan_scatter.png"
    plt.tight_layout()
    plt.savefig(png2, dpi=300)
    plt.close()
    print(f"✓ Saved scatter plot: {png2}")


# --- 报告与主流程 ----------------------------------------------------

def summarize_for_seasons(df, seasons, out_csv='MixedPlan_ControversialSeasons_summary.csv'):
    rows = []
    for s in seasons:
        s_df = df[df['Season'] == s]
        mean_gap = s_df['Judge_Fan_Gap'].mean()
        mean_rankdiff_jf = s_df['RankDiff_Judge_Fan'].mean()
        mean_rankdiff_em = s_df['RankDiff_Existing_Mixed'].mean()
        elim_changes = 0
        total_weeks = 0
        for w in s_df['Week'].unique():
            week_df = s_df[s_df['Week'] == w]
            if len(week_df) < 2:
                continue
            lowest_existing = week_df.loc[week_df['Existing_Combined'].idxmin(), 'Celebrity']
            lowest_mixed = week_df.loc[week_df['Mixed_Score'].idxmin(), 'Celebrity']
            if lowest_existing != lowest_mixed:
                elim_changes += 1
            total_weeks += 1
        rows.append({
            'Season': s,
            'Mean_Judge_Fan_Gap': mean_gap,
            'Mean_RankDiff_Judge_Fan': mean_rankdiff_jf,
            'Mean_RankDiff_Existing_Mixed': mean_rankdiff_em,
            'WeeksCompared': total_weeks,
            'EliminationDiffWeeks': elim_changes
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"✓ Exported summary CSV: {out_csv}")
    return out_df


def main(top_n=3):
    df = load_data()
    df = compute_plan3_storytelling(df)
    df = compute_rankings(df)

    controversial, season_gap = identify_controversial_seasons(df, top_n=top_n)
    print("Controversy level (avg Judge_Fan_Gap):")
    print(season_gap.head(10))
    print(f"\nSelected controversial seasons: {controversial}")

    # 绘制总览图
    plot_overview_for_seasons(df, controversial)

    # 针对每个争议赛季绘制细节图
    for s in controversial:
        plot_detailed_season(df, s)

    # Export summary
    summary = summarize_for_seasons(df, controversial)
    print('\n=== Summary: Effect of Mixed Plan on Controversial Seasons ===')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main(top_n=3)
