import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class FactorAnalysis:
    """分析舞者/明星特征对评委分与观众投票的影响（增强版）"""

    def __init__(self, data_dir: Path, out_dir: Path):
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # 加载数据（带异常处理）
        try:
            self.weekly = pd.read_csv(self.data_dir / "Weekly_Performance.csv")
            self.features = pd.read_csv(self.data_dir / "Contestant_Features.csv")
            self.estimated_fan = pd.read_csv(self.data_dir / "Estimated_Fan_Votes_Final_Model_2.csv")
            self.raw_data = pd.read_csv(self.data_dir / "2026_MCM_Problem_C_Data.csv")
        except Exception as e:
            logging.error("加载数据失败: %s", e)
            raise

    def extract_dancer_features(self):
        """提取舞者特征"""
        dancer_stats = {}
        if 'ballroom_partner' not in self.raw_data.columns:
            return pd.DataFrame(dancer_stats).T
        for dancer in self.raw_data['ballroom_partner'].dropna().unique():
            dancer_data = self.raw_data[self.raw_data['ballroom_partner'] == dancer]
            seasons_count = int(dancer_data['season'].nunique()) if 'season' in dancer_data.columns else 0
            avg_placement = dancer_data['placement'].mean() if 'placement' in dancer_data.columns else np.nan
            dancer_stats[dancer] = {
                'seasons_count': seasons_count,
                'avg_placement': avg_placement,
                'is_allstar': seasons_count >= 2,
                'experience_level': min(seasons_count, 3)
            }
        return pd.DataFrame(dancer_stats).T

    def create_merged_dataset(self):
        """合并所有数据集并做基本特征工程"""
        raw_info_cols = ['celebrity_name', 'ballroom_partner', 'celebrity_industry', 'celebrity_age_during_season', 'season']
        raw_info = self.raw_data[raw_info_cols].drop_duplicates() if set(raw_info_cols).issubset(self.raw_data.columns) else pd.DataFrame()
        merged = self.weekly.merge(
            self.estimated_fan,
            on=['Season', 'Week', 'Celebrity'],
            how='left'
        )
        if not raw_info.empty:
            merged = merged.merge(
                raw_info,
                left_on=['Celebrity', 'Season'],
                right_on=['celebrity_name', 'season'],
                how='left'
            )
        dancer_features = self.extract_dancer_features()
        # map dancer
        if 'celebrity_name' in self.raw_data.columns and 'ballroom_partner' in self.raw_data.columns:
            merged['Dancer'] = merged['Celebrity'].map(
                self.raw_data.set_index('celebrity_name')['ballroom_partner'].to_dict()
            )
        else:
            merged['Dancer'] = np.nan
        merged = merged.merge(
            dancer_features,
            left_on='Dancer',
            right_index=True,
            how='left'
        )
        merged['Age'] = merged.get('celebrity_age_during_season', merged.get('Age', np.nan))
        merged['Age_Squared'] = merged['Age'] ** 2
        merged['Industry'] = merged.get('celebrity_industry', merged.get('Industry', np.nan))
        # 保证列名一致
        if 'Estimated_Fan_Pct' not in merged.columns and 'Estimated_Fan' in merged.columns:
            merged['Estimated_Fan_Pct'] = merged['Estimated_Fan']
        if 'Judge_Pct' not in merged.columns and 'Judge_Pct_Raw' in merged.columns:
            merged['Judge_Pct'] = merged['Judge_Pct_Raw']
        merged['Judge_Fan_Diff'] = merged['Judge_Pct'] - merged['Estimated_Fan_Pct']
        merged['Week_Progress'] = merged.groupby(['Season', 'Celebrity'])['Week'].transform(
            lambda x: x / x.max() if x.max() > 0 else 0
        )
        industry_mapping = {ind: i for i, ind in enumerate(merged['Industry'].fillna('Unknown').unique())}
        merged['Industry_Code'] = merged['Industry'].map(industry_mapping).fillna(-1).astype(int)
        # 保证所需列存在且不全为空
        required = ['Judge_Pct', 'Estimated_Fan_Pct', 'Age', 'Industry_Code', 'experience_level']
        available_required = [c for c in required if c in merged.columns]
        merged = merged.dropna(subset=available_required)
        logging.info("合并后记录数: %d", len(merged))
        return merged

    def analyze_rf_with_bootstrap(self, X, y, n_boot=100, random_state=42):
        """训练随机森林并用 bootstrap 估计特征重要性置信区间"""
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state, max_depth=10)
        rf.fit(X, y)
        base_importance = rf.feature_importances_
        boot_importances = []
        rng = np.random.RandomState(random_state)
        for i in range(n_boot):
            idx = rng.randint(0, X.shape[0], X.shape[0])
            Xb, yb = X[idx], y[idx]
            rfb = RandomForestRegressor(n_estimators=100, random_state=random_state + i, max_depth=10)
            rfb.fit(Xb, yb)
            boot_importances.append(rfb.feature_importances_)
        boot = np.array(boot_importances)
        imp_mean = boot.mean(axis=0)
        imp_std = boot.std(axis=0)
        ci_lower = np.percentile(boot, 2.5, axis=0)
        ci_upper = np.percentile(boot, 97.5, axis=0)
        return rf, base_importance, imp_mean, imp_std, ci_lower, ci_upper

    def analyze_judge_score_drivers(self, data):
        """评委评分驱动因素（含 bootstrap CI）"""
        features = ['Age', 'Age_Squared', 'is_allstar', 'experience_level', 'Week_Progress', 'Industry_Code']
        X = data[features].fillna(0).values
        y = data['Judge_Pct'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf_judge, base_imp, imp_mean, imp_std, ci_l, ci_u = self.analyze_rf_with_bootstrap(X_scaled, y)
        importance_judge = pd.DataFrame({
            'Feature': features,
            'Importance': base_imp,
            'Imp_Mean': imp_mean,
            'Imp_STD': imp_std,
            'CI_Lower': ci_l,
            'CI_Upper': ci_u
        }).sort_values('Imp_Mean', ascending=False)
        return rf_judge, importance_judge

    def analyze_fan_vote_drivers(self, data):
        """观众投票驱动因素（含 bootstrap CI）"""
        features = ['Age', 'Age_Squared', 'is_allstar', 'experience_level', 'Week_Progress', 'Industry_Code']
        X = data[features].fillna(0).values
        y = data['Estimated_Fan_Pct'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf_fan, base_imp, imp_mean, imp_std, ci_l, ci_u = self.analyze_rf_with_bootstrap(X_scaled, y)
        importance_fan = pd.DataFrame({
            'Feature': features,
            'Importance': base_imp,
            'Imp_Mean': imp_mean,
            'Imp_STD': imp_std,
            'CI_Lower': ci_l,
            'CI_Upper': ci_u
        }).sort_values('Imp_Mean', ascending=False)
        return rf_fan, importance_fan

    def consistency_analysis(self, data):
        """一致性分析（总体与按赛季）"""
        results = {}
        judge_age_corr, judge_age_p = spearmanr(data['Age'].fillna(0), data['Judge_Pct'])
        fan_age_corr, fan_age_p = spearmanr(data['Age'].fillna(0), data['Estimated_Fan_Pct'])
        results['Age_Effect'] = {
            'Judge_Corr': float(judge_age_corr),
            'Judge_Pval': float(judge_age_p),
            'Fan_Corr': float(fan_age_corr),
            'Fan_Pval': float(fan_age_p),
            'Consistency': 'YES' if np.sign(judge_age_corr) == np.sign(fan_age_corr) else 'NO'
        }
        allstar_data = data.groupby('is_allstar').agg({
            'Judge_Pct': 'mean',
            'Estimated_Fan_Pct': 'mean'
        })
        if True in allstar_data.index and False in allstar_data.index:
            judge_diff = float(allstar_data.loc[True, 'Judge_Pct'] - allstar_data.loc[False, 'Judge_Pct'])
            fan_diff = float(allstar_data.loc[True, 'Estimated_Fan_Pct'] - allstar_data.loc[False, 'Estimated_Fan_Pct'])
            results['AllStar_Effect'] = {
                'Judge_Bonus': judge_diff,
                'Fan_Bonus': fan_diff,
                'Consistency': 'YES' if np.sign(judge_diff) == np.sign(fan_diff) else 'NO'
            }
        # 按赛季一致性导出
        season_rows = []
        for season, grp in data.groupby('Season'):
            ja, _ = spearmanr(grp['Age'].fillna(0), grp['Judge_Pct']) if len(grp) > 2 else (np.nan, np.nan)
            fa, _ = spearmanr(grp['Age'].fillna(0), grp['Estimated_Fan_Pct']) if len(grp) > 2 else (np.nan, np.nan)
            season_rows.append({'Season': season, 'Judge_Age_Corr': ja, 'Fan_Age_Corr': fa})
        season_df = pd.DataFrame(season_rows).sort_values('Season')
        season_df.to_csv(self.out_dir / 'Consistency_By_Season.csv', index=False)
        logging.info("已保存: %s", str(self.out_dir / 'Consistency_By_Season.csv'))
        return results

    def plot_comparison(self, importance_judge, importance_fan):
        """绘制对比图并保存 CSV"""
        sns.set(style='whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        importance_judge = importance_judge.sort_values('Imp_Mean')
        importance_fan = importance_fan.sort_values('Imp_Mean')
        importance_judge.plot(
            x='Feature', y='Imp_Mean', kind='barh', ax=axes[0], color='steelblue', legend=False, xerr=importance_judge['Imp_STD']
        )
        axes[0].set_title('Judge Score - Feature Importance (mean ± std)')
        axes[0].set_xlabel('Importance')
        importance_fan.plot(
            x='Feature', y='Imp_Mean', kind='barh', ax=axes[1], color='coral', legend=False, xerr=importance_fan['Imp_STD']
        )
        axes[1].set_title('Fan Score - Feature Importance (mean ± std)')
        axes[1].set_xlabel('Importance')
        plt.tight_layout()
        out_png = self.out_dir / '09_Factor_Comparison.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("已保存: %s", out_png)
        # 导出 CSV
        importance_judge.sort_values('Imp_Mean', ascending=False).to_csv(self.out_dir / 'Feature_Importance_Judge.csv', index=False)
        importance_fan.sort_values('Imp_Mean', ascending=False).to_csv(self.out_dir / 'Feature_Importance_Fan.csv', index=False)
        logging.info("已保存: %s, %s", str(self.out_dir / 'Feature_Importance_Judge.csv'), str(self.out_dir / 'Feature_Importance_Fan.csv'))

    def plot_feature_importance_table(self, importance_judge, importance_fan, top_n=6):
        """生成评委与观众特征重要性对比表，保存 CSV 与 PNG（表格图片）"""
        ij = importance_judge[['Feature', 'Imp_Mean', 'CI_Lower', 'CI_Upper']].rename(
            columns={'Imp_Mean': 'Imp_Mean_Judge', 'CI_Lower': 'CI_Lower_Judge', 'CI_Upper': 'CI_Upper_Judge'}
        )
        if 'Feature' not in importance_fan.columns:
            raise ValueError("importance_fan 缺少 Feature 列")
        if 'Feature' not in ij.columns:
            raise ValueError("importance_judge 缺少 Feature 列")
        if 'Imp_Mean' in importance_fan.columns:
            if_cols = ['Feature', 'Imp_Mean', 'CI_Lower', 'CI_Upper']
            if_df = importance_fan[if_cols].rename(
                columns={'Imp_Mean': 'Imp_Mean_Fan', 'CI_Lower': 'CI_Lower_Fan', 'CI_Upper': 'CI_Upper_Fan'}
            )
        else:
            raise ValueError("importance_fan 缺少 Imp_Mean 列")
        comp = ij.merge(if_df, on='Feature', how='outer').fillna(0)
        comp['Rank_Judge'] = comp['Imp_Mean_Judge'].rank(ascending=False, method='min').astype(int)
        comp['Rank_Fan'] = comp['Imp_Mean_Fan'].rank(ascending=False, method='min').astype(int)
        comp = comp.sort_values(['Rank_Judge', 'Rank_Fan'])
        # 保存 CSV
        comp.to_csv(self.out_dir / 'Feature_Importance_Compare.csv', index=False)
        logging.info("已保存: %s", str(self.out_dir / 'Feature_Importance_Compare.csv'))
        # 生成表格图片（只显示 top_n by judge importance）
        top = comp.sort_values('Imp_Mean_Judge', ascending=False).head(top_n)
        col_labels = ['Feature', 'Judge_Imp', 'Judge_CI', 'Fan_Imp', 'Fan_CI', 'Rank_Judge', 'Rank_Fan']
        cell_text = []
        for _, r in top.iterrows():
            judge_ci = f"{r['CI_Lower_Judge']:.3f}-{r['CI_Upper_Judge']:.3f}" if (r['CI_Lower_Judge'] != 0 or r['CI_Upper_Judge'] != 0) else ""
            fan_ci = f"{r['CI_Lower_Fan']:.3f}-{r['CI_Upper_Fan']:.3f}" if (r['CI_Lower_Fan'] != 0 or r['CI_Upper_Fan'] != 0) else ""
            cell_text.append([
                r['Feature'],
                f"{r['Imp_Mean_Judge']:.4f}",
                judge_ci,
                f"{r['Imp_Mean_Fan']:.4f}",
                fan_ci,
                str(r['Rank_Judge']),
                str(r['Rank_Fan'])
            ])
        fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.5), max(2, top_n * 0.6)))
        ax.axis('off')
        table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)
        out_table_png = self.out_dir / 'Feature_Importance_Compare_Table.png'
        plt.tight_layout()
        plt.savefig(out_table_png, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("已保存: %s", out_table_png)

    def generate_report(self):
        """生成分析报告并导出结果"""
        logging.info("开始第三题：特征影响因素分析")
        data = self.create_merged_dataset()
        logging.info("开始训练评委模型")
        rf_judge, importance_judge = self.analyze_judge_score_drivers(data)
        logging.info("评委特征重要性：\n%s", importance_judge.to_string(index=False))
        logging.info("开始训练观众模型")
        rf_fan, importance_fan = self.analyze_fan_vote_drivers(data)
        logging.info("观众特征重要性：\n%s", importance_fan.to_string(index=False))
        consistency = self.consistency_analysis(data)
        logging.info("一致性分析结果：%s", consistency)
        self.plot_comparison(importance_judge, importance_fan)
        # 新增：导出并绘制评委 vs 观众 特征重要性对比表
        self.plot_feature_importance_table(importance_judge, importance_fan, top_n=6)
        # 保存模型（可选）
        try:
            import joblib
            joblib.dump(rf_judge, self.out_dir / 'rf_judge.joblib')
            joblib.dump(rf_fan, self.out_dir / 'rf_fan.joblib')
            logging.info("模型已保存到 %s", str(self.out_dir))
        except Exception:
            logging.debug("joblib 未安装或保存失败，已跳过模型保存。")
        logging.info("分析完成，输出保存在：%s", str(self.out_dir))


def main():
    parser = argparse.ArgumentParser(description="Question 3: Factor Analysis (增强版)")
    parser.add_argument("--data-dir", default=r"d:\MCM-code", help="数据目录（默认 d:\\MCM-code）")
    parser.add_argument("--out-dir", default=r"d:\MCM-code\fig", help="输出目录（默认 d:\\MCM-code\\fig）")
    args = parser.parse_args()
    fa = FactorAnalysis(Path(args.data_dir), Path(args.out_dir))
    fa.generate_report()


if __name__ == '__main__':
    main()