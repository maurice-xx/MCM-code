import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Set English fonts and styling
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def validate_and_visualize():
    """验证粉丝投票估计值并生成可视化"""
    
    # 1. Load data
    try:
        estimated = pd.read_csv("Estimated_Fan_Votes_Final_Model_1.csv")
        weekly = pd.read_csv("Weekly_Performance.csv")
        elim = pd.read_csv("Elimination_Lookup.csv")
        features = pd.read_csv("Contestant_Features.csv")
    except FileNotFoundError as e:
        print(f"❌ Data file missing: {e}")
        return
    
    print("=" * 80)
    print("FAN VOTE ESTIMATION MODEL VALIDATION & VISUALIZATION")
    print("=" * 80)
    
    # 2. Add rule information
    def get_rule(season):
        return "Rank-Based" if season <= 2 else "Percentage-Based"
    
    estimated['Rule'] = estimated['Season'].apply(get_rule)
    weekly['Rule'] = weekly['Season'].apply(get_rule)
    
    # =====================================================================
    # PART 1: CONSISTENCY METRICS (一致性指标)
    # =====================================================================
    
    print("\n" + "="*80)
    print("PART 1: CONSISTENCY METRICS")
    print("="*80)
    
    consistency_results = {
        'Rank-Based': {'valid_weeks': 0, 'total_weeks': 0, 'accuracy': 0},
        'Percentage-Based': {'valid_weeks': 0, 'total_weeks': 0, 'accuracy': 0}
    }
    
    # 验证 Rank-Based (Season 1-2)
    print("\n【Rank-Based Rule Validation】(Season 1-2)")
    print("-" * 80)
    rank_data = estimated[estimated['Rule'] == 'Rank-Based']
    
    if len(rank_data) > 0:
        rank_by_week = rank_data.groupby(['Season', 'Week'])
        valid_rank_count = 0
        total_weeks = 0
        correct_elim_rank = 0
        total_elim_rank = 0
        
        for (season, week), group in rank_by_week:
            total_weeks += 1
            n_contestants = len(group)
            ranks = sorted(group['Estimated_Fan_Rank'].dropna().values)
            expected_ranks = list(range(1, n_contestants + 1))
            
            if ranks == expected_ranks:
                valid_rank_count += 1
            
            # Check elimination logic
            judge_data = weekly[(weekly['Season'] == season) & (weekly['Week'] == week)]
            elim_row = elim[(elim['season'] == season) & (elim['Elim_Week'] == week)]
            
            if len(elim_row) > 0 and len(judge_data) > 0:
                elim_celebrity = elim_row.iloc[0]['celebrity_name']
                total_elim_rank += 1
                
                week_combined = []
                for _, r in group.iterrows():
                    c = r['Celebrity']
                    j_row = judge_data[judge_data['Celebrity'] == c]
                    if len(j_row) > 0:
                        j_rank = j_row.iloc[0]['Judge_Rank']
                        f_rank = r['Estimated_Fan_Rank']
                        if pd.notna(j_rank) and pd.notna(f_rank):
                            week_combined.append((c, j_rank + f_rank))
                
                if week_combined:
                    max_celebrity = max(week_combined, key=lambda x: x[1])[0]
                    if max_celebrity == elim_celebrity:
                        correct_elim_rank += 1
        
        print(f"✓ Valid Ranking Weeks: {valid_rank_count}/{total_weeks} ({100*valid_rank_count/total_weeks:.1f}%)")
        if total_elim_rank > 0:
            accuracy_rank = 100 * correct_elim_rank / total_elim_rank
            print(f"✓ Elimination Prediction Accuracy: {correct_elim_rank}/{total_elim_rank} = {accuracy_rank:.1f}%")
            consistency_results['Rank-Based']['accuracy'] = accuracy_rank
        
        consistency_results['Rank-Based']['valid_weeks'] = valid_rank_count
        consistency_results['Rank-Based']['total_weeks'] = total_weeks
    else:
        print("(No Rank-Based data)")
    
    # 验证 Percentage-Based (Season 3+)
    print("\n【Percentage-Based Rule Validation】(Season 3+)")
    print("-" * 80)
    percent_data = estimated[estimated['Rule'] == 'Percentage-Based']
    
    if len(percent_data) > 0:
        percent_by_week = percent_data.groupby(['Season', 'Week'])
        sum_to_one_count = 0
        total_weeks_percent = 0
        correct_elim_percent = 0
        total_elim_percent = 0
        
        for (season, week), group in percent_by_week:
            total_weeks_percent += 1
            pct_sum = group['Estimated_Fan_Pct'].sum()
            
            if abs(pct_sum - 1.0) < 0.01:
                sum_to_one_count += 1
            
            # Check elimination logic
            judge_data = weekly[(weekly['Season'] == season) & (weekly['Week'] == week)]
            elim_row = elim[(elim['season'] == season) & (elim['Elim_Week'] == week)]
            
            if len(elim_row) > 0 and len(judge_data) > 0:
                elim_celebrity = elim_row.iloc[0]['celebrity_name']
                total_elim_percent += 1
                
                week_combined_pct = []
                for _, r in group.iterrows():
                    c = r['Celebrity']
                    j_row = judge_data[judge_data['Celebrity'] == c]
                    if len(j_row) > 0:
                        j_pct = j_row.iloc[0]['Judge_Pct']
                        f_pct = r['Estimated_Fan_Pct']
                        if pd.notna(j_pct) and pd.notna(f_pct):
                            week_combined_pct.append((c, j_pct + f_pct))
                
                if week_combined_pct:
                    min_celebrity = min(week_combined_pct, key=lambda x: x[1])[0]
                    if min_celebrity == elim_celebrity:
                        correct_elim_percent += 1
        
        print(f"✓ Valid Percentage-Summing Weeks: {sum_to_one_count}/{total_weeks_percent} ({100*sum_to_one_count/total_weeks_percent:.1f}%)")
        if total_elim_percent > 0:
            accuracy_percent = 100 * correct_elim_percent / total_elim_percent
            print(f"✓ Elimination Prediction Accuracy: {correct_elim_percent}/{total_elim_percent} = {accuracy_percent:.1f}%")
            consistency_results['Percentage-Based']['accuracy'] = accuracy_percent
        
        consistency_results['Percentage-Based']['valid_weeks'] = sum_to_one_count
        consistency_results['Percentage-Based']['total_weeks'] = total_weeks_percent
    else:
        print("(No Percentage-Based data)")
    
    # =====================================================================
    # PART 2: UNCERTAINTY METRICS (不确定性度量)
    # =====================================================================
    
    print("\n" + "="*80)
    print("PART 2: UNCERTAINTY METRICS")
    print("="*80)
    
    print(f"\n✓ Fan Vote Certainty (Std Dev) Statistics:")
    print(f"   Mean: {estimated['Certainty_Std'].mean():.4f}")
    print(f"   Median: {estimated['Certainty_Std'].median():.4f}")
    print(f"   Std Dev: {estimated['Certainty_Std'].std():.4f}")
    print(f"   Range: [{estimated['Certainty_Std'].min():.4f}, {estimated['Certainty_Std'].max():.4f}]")
    
    # Uncertainty by season
    print(f"\n✓ Uncertainty by Season:")
    season_uncertainty = estimated.groupby('Season')['Certainty_Std'].agg(['mean', 'std', 'min', 'max'])
    print(season_uncertainty)
    
    # Uncertainty by rule
    print(f"\n✓ Uncertainty by Rule:")
    rule_uncertainty = estimated.groupby('Rule')['Certainty_Std'].agg(['mean', 'std', 'min', 'max'])
    print(rule_uncertainty)
    
    # =====================================================================
    # VISUALIZATION 1: Consistency Metrics Dashboard
    # =====================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Consistency & Reliability Metrics', fontsize=16, fontweight='bold')
    
    # 1a. Accuracy by Rule
    rules = list(consistency_results.keys())
    accuracies = [consistency_results[r]['accuracy'] for r in rules]
    colors = ['#2E86AB', '#A23B72']
    
    ax = axes[0, 0]
    bars = ax.bar(rules, accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Elimination Prediction Accuracy by Rule Type', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Baseline')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1b. Valid Weeks Coverage
    ax = axes[0, 1]
    coverage_data = [
        (consistency_results[r]['valid_weeks'], consistency_results[r]['total_weeks'])
        for r in rules
    ]
    valid_weeks = [c[0] for c in coverage_data]
    total_weeks = [c[1] for c in coverage_data]
    
    x = np.arange(len(rules))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, valid_weeks, width, label='Valid Weeks', 
                   color='#06A77D', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, total_weeks, width, label='Total Weeks', 
                   color='#D62828', edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Number of Weeks', fontsize=11, fontweight='bold')
    ax.set_title('Valid Weeks Coverage by Rule Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rules)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1c. Uncertainty Distribution
    ax = axes[1, 0]
    for rule, color in zip(rules, colors):
        data = estimated[estimated['Rule'] == rule]['Certainty_Std'].dropna()
        ax.hist(data, bins=30, alpha=0.6, label=rule, color=color, edgecolor='black')
    
    ax.set_xlabel('Certainty Std Dev', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Prediction Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1d. Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Rank-Based', 'Percentage-Based'],
        ['Accuracy', f"{consistency_results['Rank-Based']['accuracy']:.1f}%", 
         f"{consistency_results['Percentage-Based']['accuracy']:.1f}%"],
        ['Valid Weeks', f"{consistency_results['Rank-Based']['valid_weeks']}/{consistency_results['Rank-Based']['total_weeks']}",
         f"{consistency_results['Percentage-Based']['valid_weeks']}/{consistency_results['Percentage-Based']['total_weeks']}"],
        ['Avg Uncertainty', f"{estimated[estimated['Rule']=='Rank-Based']['Certainty_Std'].mean():.4f}",
         f"{estimated[estimated['Rule']=='Percentage-Based']['Certainty_Std'].mean():.4f}"],
        ['Total Records', f"{len(estimated[estimated['Rule']=='Rank-Based'])}", 
         f"{len(estimated[estimated['Rule']=='Percentage-Based'])}"]
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.tight_layout()
    plt.savefig('01_Consistency_Metrics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 01_Consistency_Metrics.png")
    plt.close()
    
    # =====================================================================
    # VISUALIZATION 2: Uncertainty Analysis by Season
    # =====================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Uncertainty Analysis by Season', fontsize=16, fontweight='bold')
    
    # 2a. Box plot by season
    ax = axes[0, 0]
    season_data = [estimated[estimated['Season'] == s]['Certainty_Std'].dropna() 
                   for s in sorted(estimated['Season'].unique())]
    bp = ax.boxplot(season_data, labels=sorted(estimated['Season'].unique()),
                    patch_artist=True, widths=0.6)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#A23B72')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Season', fontsize=11, fontweight='bold')
    ax.set_ylabel('Certainty Std Dev', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Uncertainty Distribution by Season', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2b. Mean uncertainty trend by season
    ax = axes[0, 1]
    season_means = estimated.groupby('Season')['Certainty_Std'].mean().sort_index()
    season_stds = estimated.groupby('Season')['Certainty_Std'].std().sort_index()
    
    ax.errorbar(season_means.index, season_means.values, yerr=season_stds.values,
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                color='#2E86AB', ecolor='#E74C3C', alpha=0.7, label='Mean ± Std Dev')
    
    ax.set_xlabel('Season', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Certainty Std Dev', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty Trend Across Seasons', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2c. Heatmap: Uncertainty by Season & Week (subset for clarity)
    ax = axes[1, 0]
    pivot_data = estimated.groupby(['Season', 'Week'])['Certainty_Std'].mean().unstack()
    
    # Show only first 5 seasons and weeks 1-8
    pivot_subset = pivot_data.loc[pivot_data.index.isin(range(1, 6)), 
                                  pivot_data.columns.isin(range(1, 9))]
    
    im = ax.imshow(pivot_subset, cmap='RdYlGn_r', aspect='auto')
    ax.set_xlabel('Week', fontsize=11, fontweight='bold')
    ax.set_ylabel('Season', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty Heatmap: Seasons 1-5, Weeks 1-8', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(pivot_subset.columns)))
    ax.set_xticklabels(pivot_subset.columns)
    ax.set_yticks(range(len(pivot_subset.index)))
    ax.set_yticklabels(pivot_subset.index)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Std Dev', fontweight='bold')
    
    # 2d. Violin plot: Rule comparison
    ax = axes[1, 1]
    rule_data = [estimated[estimated['Rule'] == 'Rank-Based']['Certainty_Std'].dropna(),
                 estimated[estimated['Rule'] == 'Percentage-Based']['Certainty_Std'].dropna()]
    
    parts = ax.violinplot(rule_data, positions=[1, 2], showmeans=True, showmedians=True)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Rank-Based', 'Percentage-Based'])
    ax.set_ylabel('Certainty Std Dev', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty Distribution by Rule Type', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_Uncertainty_by_Season.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_Uncertainty_by_Season.png")
    plt.close()
    
    # =====================================================================
    # VISUALIZATION 3: Model Fit Quality
    # =====================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Fit Quality Assessment', fontsize=16, fontweight='bold')
    
    # 3a. Fit Status Distribution
    ax = axes[0, 0]
    fit_counts = estimated['Fit_Status'].value_counts()
    colors_fit = ['#06A77D' if x == 'Converged' else '#D62828' for x in fit_counts.index]
    
    wedges, texts, autotexts = ax.pie(fit_counts.values, labels=fit_counts.index,
                                        autopct='%1.1f%%', colors=colors_fit,
                                        startangle=90, explode=(0.05, 0.05),
                                        textprops={'fontsize': 11, 'weight': 'bold'})
    
    ax.set_title('Distribution of Model Fit Status', fontsize=12, fontweight='bold')
    
    # 3b. Alpha & Beta Parameters Distribution
    ax = axes[0, 1]
    
    alpha_values = estimated['Model_Alpha'].unique()
    beta_values = estimated['Model_Beta'].unique()
    
    ax.scatter(estimated['Model_Alpha'], estimated['Model_Beta'], 
              alpha=0.6, s=100, c=estimated['Certainty_Std'], 
              cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Alpha (Popularity Weight)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Beta (Judge Influence Weight)', fontsize=11, fontweight='bold')
    ax.set_title('Model Parameter Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Uncertainty', fontweight='bold')
    
    # 3c. Convergence Quality
    ax = axes[1, 0]
    
    fit_status_by_season = pd.crosstab(estimated['Season'], estimated['Fit_Status'], normalize='index') * 100
    
    fit_status_by_season.plot(kind='bar', ax=ax, color=['#06A77D', '#D62828'],
                              edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax.set_xlabel('Season', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Model Convergence by Season', fontsize=12, fontweight='bold')
    ax.legend(title='Fit Status', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    # 3d. Residual Statistics
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    convergence_rate = (estimated['Fit_Status'] == 'Converged').sum() / len(estimated) * 100
    
    quality_data = [
        ['Metric', 'Value'],
        ['Total Records', f"{len(estimated)}"],
        ['Converged Models', f"{(estimated['Fit_Status'] == 'Converged').sum()} ({convergence_rate:.1f}%)"],
        ['Failed-to-Fit Models', f"{(estimated['Fit_Status'] == 'Failed_to_Fit').sum()}"],
        ['Avg Alpha', f"{estimated['Model_Alpha'].mean():.4f}"],
        ['Avg Beta', f"{estimated['Model_Beta'].mean():.4f}"],
        ['Uncertainty Range', f"[{estimated['Certainty_Std'].min():.4f}, {estimated['Certainty_Std'].max():.4f}]"],
        ['Coverage (Valid Weeks)', f"{sum([consistency_results[r]['valid_weeks'] for r in rules])}/{sum([consistency_results[r]['total_weeks'] for r in rules])}"]
    ]
    
    table = ax.table(cellText=quality_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(quality_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.tight_layout()
    plt.savefig('03_Model_Fit_Quality.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_Model_Fit_Quality.png")
    plt.close()
    
    # =====================================================================
    # VISUALIZATION 4: Confidence Intervals & Reliability
    # =====================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Confidence & Reliability Analysis', fontsize=16, fontweight='bold')
    
    # 4a. Confidence by Contestant Popularity
    ax = axes[0, 0]
    estimated_merged = estimated.merge(
        features[['Name', 'Partner_Strength_Index']], 
        left_on='Celebrity', right_on='Name', how='left'
    )
    
    scatter = ax.scatter(estimated_merged['Partner_Strength_Index'], 
                        estimated_merged['Certainty_Std'],
                        alpha=0.5, s=80, c=estimated_merged['Season'],
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Partner Strength Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Uncertainty', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty vs. Performer Characteristics', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Season', fontweight='bold')
    
    # 4b. Uncertainty by Week Number
    ax = axes[0, 1]
    week_stats = estimated.groupby('Week')['Certainty_Std'].agg(['mean', 'std']).reset_index()
    week_stats = week_stats[week_stats['Week'] <= 15]  # Show first 15 weeks
    
    ax.fill_between(week_stats['Week'], 
                     week_stats['mean'] - week_stats['std'],
                     week_stats['mean'] + week_stats['std'],
                     alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    ax.plot(week_stats['Week'], week_stats['mean'], 
           'o-', linewidth=2, markersize=8, color='#2E86AB', label='Mean Uncertainty')
    
    ax.set_xlabel('Week Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Certainty Std Dev', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty Trend Throughout Competition', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4c. QQ Plot (Normal Distribution Check)
    ax = axes[1, 0]
    from scipy import stats
    
    std_dev_data = estimated['Certainty_Std'].dropna()
    stats.probplot(std_dev_data, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Normality of Uncertainty Distribution', 
                fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4d. Cumulative Distribution
    ax = axes[1, 1]
    
    for rule, color in zip(['Rank-Based', 'Percentage-Based'], ['#2E86AB', '#A23B72']):
        data = estimated[estimated['Rule'] == rule]['Certainty_Std'].dropna().values
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cumulative, linewidth=2.5, label=rule, color=color, marker='o', 
               markersize=4, markevery=max(1, len(sorted_data)//20))
    
    ax.set_xlabel('Certainty Std Dev', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Distribution of Uncertainty', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_Confidence_Analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_Confidence_Analysis.png")
    plt.close()
    
    # =====================================================================
    # SUMMARY REPORT
    # =====================================================================
    
    print("\n" + "="*80)
    print("SUMMARY & KEY FINDINGS")
    print("="*80)
    
    print(f"\n📊 CONSISTENCY INDICATORS:")
    print(f"   • Rank-Based Accuracy: {consistency_results['Rank-Based']['accuracy']:.1f}%")
    print(f"   • Percentage-Based Accuracy: {consistency_results['Percentage-Based']['accuracy']:.1f}%")
    print(f"   • Overall Coverage: {consistency_results['Rank-Based']['valid_weeks'] + consistency_results['Percentage-Based']['valid_weeks']}/"\
          f"{consistency_results['Rank-Based']['total_weeks'] + consistency_results['Percentage-Based']['total_weeks']}")
    
    print(f"\n📈 UNCERTAINTY INDICATORS:")
    print(f"   • Mean Uncertainty: {estimated['Certainty_Std'].mean():.4f}")
    print(f"   • Median Uncertainty: {estimated['Certainty_Std'].median():.4f}")
    print(f"   • Uncertainty Coefficient of Variation: {estimated['Certainty_Std'].std() / estimated['Certainty_Std'].mean():.2f}")
    
    print(f"\n✅ CONFIDENCE LEVEL:")
    convergence_pct = (estimated['Fit_Status'] == 'Converged').sum() / len(estimated) * 100
    print(f"   • Model Convergence Rate: {convergence_pct:.1f}%")
    print(f"   • Recommendation: {'HIGH' if convergence_pct > 80 and consistency_results['Rank-Based']['accuracy'] > 60 else 'MEDIUM' if convergence_pct > 60 else 'LOW'}")
    
    print("\n✓ All visualizations saved successfully!")
    print("\n" + "="*80)

if __name__ == "__main__":
    validate_and_visualize()