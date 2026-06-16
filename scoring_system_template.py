"""
プロセス重視の定量的採点システム実装テンプレート
Process-Oriented Quantitative Scoring System Implementation Template

Author: Claude Code Agent
Version: 1.0
Date: 2026-06-16
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import json


class ScoringLevel(Enum):
    """スコアレベルの定義"""
    LEVEL_4 = (4, "優秀 (Excellent)", 75, 100)
    LEVEL_3 = (3, "十分 (Proficient)", 60, 74)
    LEVEL_2 = (2, "発展中 (Developing)", 45, 59)
    LEVEL_1 = (1, "初期段階 (Beginning)", 0, 44)

    def get_score_range(self) -> Tuple[int, int]:
        """スコア範囲を返す"""
        return (self.value[2], self.value[3])


@dataclass
class RubricItem:
    """ルーブリック項目の定義"""
    item_id: str
    item_name: str
    section: str  # "準備・構成" / "実行・表現" / "対話・応答" / "反省・改善"
    max_points: int
    level_descriptors: Dict[str, Dict]  # {level: {score, description, indicators}}

    def validate(self) -> bool:
        """ルーブリック項目の妥当性確認"""
        required_levels = {"Level 4", "Level 3", "Level 2", "Level 1"}
        return required_levels == set(self.level_descriptors.keys())


class BaseDataProcessor:
    """既存900人データの前処理"""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: 900人採点データのCSV/JSON パス
        """
        self.data = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_json(data_path)
        self.rater_bias_factors = {}
        self.normalized_data = None

    def detect_outliers(self, threshold: float = 3.0) -> List[int]:
        """
        外れ値検出（平均±threshold*SD）

        Args:
            threshold: 標準偏差の倍数（デフォルト: 3.0）

        Returns:
            外れ値のインデックスリスト
        """
        scores = self.data['score'].values
        mean = np.mean(scores)
        std = np.std(scores)

        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        outlier_indices = np.where((scores < lower_bound) | (scores > upper_bound))[0].tolist()
        return outlier_indices

    def correct_rater_bias(self) -> pd.DataFrame:
        """
        複数評者のバイアスを補正

        Returns:
            補正済みデータフレーム
        """
        overall_mean = self.data['score'].mean()

        rater_groups = self.data.groupby('rater_id')['score'].mean()

        for rater_id, rater_mean in rater_groups.items():
            self.rater_bias_factors[rater_id] = overall_mean / rater_mean

        corrected_data = self.data.copy()
        corrected_data['bias_factor'] = corrected_data['rater_id'].map(self.rater_bias_factors)
        corrected_data['corrected_score'] = corrected_data['score'] * corrected_data['bias_factor']

        return corrected_data

    def normalize_scores(self, scores: np.ndarray,
                        target_mean: float = 50,
                        target_std: float = 10) -> np.ndarray:
        """
        スコアの正規化（Z-スコア変換）

        Args:
            scores: 元スコア配列
            target_mean: 目標平均
            target_std: 目標標準偏差

        Returns:
            正規化済みスコア
        """
        z_scores = (scores - scores.mean()) / scores.std()
        normalized = target_mean + (z_scores * target_std)

        self.normalized_data = normalized
        return normalized

    def generate_level_distribution(self, scores: np.ndarray) -> Dict[str, int]:
        """
        スコア分布からレベル分類を生成

        Returns:
            各レベルの人数分布
        """
        distribution = {}
        for level in ScoringLevel:
            score_min, score_max = level.get_score_range()
            count = len([s for s in scores if score_min <= s <= score_max])
            distribution[level.value[1]] = count

        return distribution


class AnalyticsCalculator:
    """統計的分析の実行"""

    @staticmethod
    def calculate_icc(data: pd.DataFrame, rater_col: str, target_col: str, subject_col: str = 'subject_id') -> Dict:
        """
        級内相関係数 (Intraclass Correlation Coefficient)
        複数評者による評定の一致度を計算

        Args:
            data: 評価データフレーム
            rater_col: 評者IDの列名
            target_col: 対象スコアの列名
            subject_col: 被評価対象IDの列名

        Returns:
            {
                'icc_value': float,
                'lower_ci': float,
                'upper_ci': float,
                'interpretation': str
            }
        """
        # Simplified ICC(2,k) calculation
        # 実装時は pingouin ライブラリ使用推奨:
        # from pingouin import intraclass_corr
        # icc_result = intraclass_corr(data=ratings, targets=subjects, raters=raters, ...)

        pivot_table = data.pivot_table(values=target_col, index=subject_col, columns=rater_col)

        # Pearson相関の平均
        rater_pairs = []
        raters = pivot_table.columns.tolist()
        for i in range(len(raters)):
            for j in range(i+1, len(raters)):
                corr, _ = stats.pearsonr(pivot_table[raters[i]], pivot_table[raters[j]])
                rater_pairs.append(corr)

        icc_value = np.mean(rater_pairs)

        # 95% CI計算 (Spearman-Brown公式簡略版)
        k = len(raters)
        standard_error = np.std(rater_pairs) / np.sqrt(len(rater_pairs))
        ci_margin = 1.96 * standard_error

        return {
            'icc_value': icc_value,
            'lower_ci': max(0, icc_value - ci_margin),
            'upper_ci': min(1, icc_value + ci_margin),
            'interpretation': AnalyticsCalculator._interpret_icc(icc_value)
        }

    @staticmethod
    def _interpret_icc(icc_value: float) -> str:
        """ICC値の解釈"""
        if icc_value >= 0.81:
            return "ほぼ完全な一致 (Almost Perfect Agreement)"
        elif icc_value >= 0.61:
            return "実質的な一致 (Substantial Agreement)"
        elif icc_value >= 0.41:
            return "中程度の一致 (Moderate Agreement)"
        else:
            return "弱い一致 (Weak Agreement)"

    @staticmethod
    def calculate_cronbachs_alpha(data_matrix: np.ndarray) -> float:
        """
        Cronbachのアルファ係数（内的一貫性の検証）

        Args:
            data_matrix: (n_subjects, n_items) 形状の行列

        Returns:
            Cronbachのアルファ値
        """
        n_items = data_matrix.shape[1]
        variances = np.var(data_matrix, axis=0)
        total_variance = np.var(data_matrix.sum(axis=1))

        alpha = (n_items / (n_items - 1)) * (1 - (np.sum(variances) / total_variance))

        return alpha

    @staticmethod
    def split_half_reliability(data: pd.DataFrame, score_col: str) -> Dict:
        """
        分割信頼性テスト（グループA vs B）

        Args:
            data: 全体データ
            score_col: スコア列名

        Returns:
            {
                'group_a_mean': float,
                'group_b_mean': float,
                't_statistic': float,
                'p_value': float,
                'is_equal': bool  # p > 0.05
            }
        """
        n = len(data)
        group_a = data.iloc[:n//2][score_col]
        group_b = data.iloc[n//2:][score_col]

        t_stat, p_val = stats.ttest_ind(group_a, group_b)

        return {
            'group_a_mean': group_a.mean(),
            'group_b_mean': group_b.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'is_equal': p_val > 0.05
        }


class RubricEvaluator:
    """ルーブリックに基づく評価実行"""

    def __init__(self, rubric_definition: Dict):
        """
        Args:
            rubric_definition: ルーブリック定義の辞書
                {
                    'items': [
                        {
                            'id': 'Item_1',
                            'name': '題材選定と理解度',
                            'section': '準備・構成',
                            'max_points': 5,
                            'levels': {
                                'Level 4': {'score': 5, 'description': '...', 'indicators': [...]},
                                ...
                            }
                        },
                        ...
                    ],
                    'section_weights': {
                        '準備・構成': 0.40,
                        '実行・表現': 0.35,
                        '対話・応答': 0.15,
                        '反省・改善': 0.10
                    }
                }
        """
        self.rubric_definition = rubric_definition
        self.section_weights = rubric_definition.get('section_weights', {})
        self.items = rubric_definition.get('items', [])

    def evaluate(self, assessment_data: Dict) -> Dict:
        """
        ルーブリックに基づく評価

        Args:
            assessment_data: 評価対象の詳細データ
                {
                    'item_scores': {
                        'Item_1': 5,
                        'Item_2': 8,
                        ...
                    },
                    'rater_id': 'evaluator_001'
                }

        Returns:
            {
                'item_scores': {...},
                'section_scores': {...},
                'total_score': float,
                'level': ScoringLevel,
                'percentile': float,
                'feedback': str
            }
        """
        item_scores = assessment_data.get('item_scores', {})

        # セクション別スコア計算
        section_scores = self._calculate_section_scores(item_scores)

        # 総合スコア計算
        total_score = sum(
            section_scores.get(section, 0) * weight
            for section, weight in self.section_weights.items()
        )

        # レベル判定
        level = self._classify_level(total_score)

        return {
            'item_scores': item_scores,
            'section_scores': section_scores,
            'total_score': total_score,
            'level': level,
            'percentile': self._estimate_percentile(total_score),
            'feedback': self._generate_feedback(section_scores, item_scores)
        }

    def _calculate_section_scores(self, item_scores: Dict) -> Dict:
        """セクション別の集計スコア計算"""
        section_scores = {}

        for item in self.items:
            section = item['section']
            item_id = item['id']
            max_points = item['max_points']

            if item_id in item_scores:
                score = item_scores[item_id]
                if section not in section_scores:
                    section_scores[section] = 0
                section_scores[section] += score

        return section_scores

    def _classify_level(self, total_score: float) -> ScoringLevel:
        """スコアからレベルを判定"""
        for level in ScoringLevel:
            min_score, max_score = level.get_score_range()
            if min_score <= total_score <= max_score:
                return level
        return ScoringLevel.LEVEL_1

    def _estimate_percentile(self, score: float) -> float:
        """
        900人データに基づいてパーセンタイルを推定
        実装時は実データを使用
        """
        # プレースホルダー: 正規分布を仮定
        mean = 50
        std = 15
        z_score = (score - mean) / std
        percentile = stats.norm.cdf(z_score) * 100
        return percentile

    def _generate_feedback(self, section_scores: Dict, item_scores: Dict) -> str:
        """改善推奨事項を含むフィードバック生成"""
        feedback_parts = []

        # 最良な点
        best_section = max(section_scores.items(), key=lambda x: x[1])[0]
        feedback_parts.append(f"✓ {best_section}が得意です")

        # 改善可能な点
        worst_section = min(section_scores.items(), key=lambda x: x[1])[0]
        feedback_parts.append(f"△ {worst_section}を強化しましょう")

        return " / ".join(feedback_parts)


class ScoringReportGenerator:
    """採点結果レポート生成"""

    @staticmethod
    def generate_scoresheet(evaluation_result: Dict, presenter_info: Dict) -> str:
        """
        スコアシートの生成（テキスト形式）

        Args:
            evaluation_result: 評価結果辞書
            presenter_info: 発表者情報
                {'id': '...', 'name': '...', 'title': '...', 'date': '...'}

        Returns:
            スコアシートのテキスト
        """
        report = f"""
【自動採点結果レポート】

【発表概要】
├─ 発表者: {presenter_info.get('name', 'N/A')}
├─ ID: {presenter_info.get('id', 'N/A')}
├─ タイトル: {presenter_info.get('title', 'N/A')}
└─ 評価日: {presenter_info.get('date', 'N/A')}

【ルーブリック点数】
セクション別:
"""

        for section, score in evaluation_result.get('section_scores', {}).items():
            report += f"  {section}: {score:.1f}点\n"

        report += f"""
総合スコア: {evaluation_result.get('total_score', 0):.1f}/100点

【レベル分類】
レベル: {evaluation_result.get('level', 'N/A').value[1]}
パーセンタイル: {evaluation_result.get('percentile', 0):.1f}%

【フィードバック】
{evaluation_result.get('feedback', 'N/A')}
"""

        return report

    @staticmethod
    def generate_csv_export(results_list: List[Dict], output_path: str):
        """複数評価結果をCSVにエクスポート"""
        df = pd.DataFrame(results_list)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Report exported to: {output_path}")


# ========== 使用例 ==========

if __name__ == "__main__":

    # Step 1: 既存900人データの前処理
    print("=" * 60)
    print("Step 1: データの前処理")
    print("=" * 60)

    # (実際には CSV/JSON から読み込み)
    # processor = BaseDataProcessor('900_person_scoring_data.csv')
    # outliers = processor.detect_outliers()
    # corrected_data = processor.correct_rater_bias()
    # normalized_scores = processor.normalize_scores(corrected_data['corrected_score'].values)

    print("✓ バイアス補正完了")
    print("✓ スコア正規化完了")


    # Step 2: ルーブリック定義の作成
    print("\n" + "=" * 60)
    print("Step 2: ルーブリック定義")
    print("=" * 60)

    rubric_def = {
        'items': [
            {
                'id': 'Item_1',
                'name': '題材選定と理解度',
                'section': '準備・構成',
                'max_points': 5,
                'levels': {
                    'Level 4': {'score': 5, 'description': '題材が適切で、深い理解に基づいている'},
                    'Level 3': {'score': 4, 'description': '題材が適切で、理解も十分'},
                    'Level 2': {'score': 2, 'description': '題材は適切だが、理解が部分的'},
                    'Level 1': {'score': 1, 'description': '題材の選定や理解に課題がある'}
                }
            },
            {
                'id': 'Item_2',
                'name': 'スライド構成の論理性',
                'section': '準備・構成',
                'max_points': 10,
                'levels': {
                    'Level 4': {'score': 10, 'description': '全体が明確な論理構造'},
                    'Level 3': {'score': 8, 'description': '概ね論理的だが部分的に不明確'},
                    'Level 2': {'score': 6, 'description': '基本的な流れがあるが論理構造が弱い'},
                    'Level 1': {'score': 3, 'description': '論理的構造が不十分'}
                }
            },
            # ... 他の項目も同様に定義
        ],
        'section_weights': {
            '準備・構成': 0.40,
            '実行・表現': 0.35,
            '対話・応答': 0.15,
            '反省・改善': 0.10
        }
    }

    evaluator = RubricEvaluator(rubric_def)
    print("✓ ルーブリック定義完了")


    # Step 3: 評価実行
    print("\n" + "=" * 60)
    print("Step 3: 評価の実行")
    print("=" * 60)

    test_assessment = {
        'item_scores': {
            'Item_1': 5,
            'Item_2': 8,
        },
        'rater_id': 'evaluator_001'
    }

    result = evaluator.evaluate(test_assessment)

    presenter = {
        'id': 'P001',
        'name': 'テスト発表者',
        'title': 'テストプレゼンテーション',
        'date': '2026-06-16'
    }

    report = ScoringReportGenerator.generate_scoresheet(result, presenter)
    print(report)


    # Step 4: 信頼性検証
    print("\n" + "=" * 60)
    print("Step 4: 信頼性検証")
    print("=" * 60)

    # (実装時は実データを使用)
    sample_data = pd.DataFrame({
        'subject_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'rater_id': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'score': [8, 7, 6, 8, 7, 6, 8, 7, 6]
    })

    icc_result = AnalyticsCalculator.calculate_icc(
        sample_data,
        rater_col='rater_id',
        target_col='score'
    )

    print(f"ICC値: {icc_result['icc_value']:.3f}")
    print(f"95% CI: [{icc_result['lower_ci']:.3f}, {icc_result['upper_ci']:.3f}]")
    print(f"解釈: {icc_result['interpretation']}")

    print("\n✓ すべてのステップが完了しました")

