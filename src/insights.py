"""
Insights Generation Module

This module generates business insights and visual trends for pharmaceutical
drug sales forecasting analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


class InsightsGenerator:
    """
    Generates business insights from pharmaceutical drug sales analysis.
    """
    
    # Default sample size for visualization performance optimization
    DEFAULT_SAMPLE_SIZE = 5000
    
    def __init__(self, output_path: str = 'outputs/', 
                 sample_size: int = DEFAULT_SAMPLE_SIZE):
        """
        Initialize insights generator.
        
        Args:
            output_path: Path to save insights and visualizations
            sample_size: Maximum sample size for visualizations (for performance)
        """
        self.output_path = output_path
        self.insights = []
        self.sample_size = sample_size
        os.makedirs(output_path, exist_ok=True)
        
    def identify_high_erosion_drugs(self, df: pd.DataFrame,
                                     threshold: float = 0.6) -> pd.DataFrame:
        """
        Identify drugs with high erosion rates.
        
        Args:
            df: DataFrame with erosion metrics
            threshold: Erosion threshold for classification (default 60%)
            
        Returns:
            DataFrame of high erosion drugs
        """
        # Get maximum erosion per drug
        post_entry = df[df['months_postgx'] >= 0]
        max_erosion = post_entry.groupby('drug_id')['erosion'].max().reset_index()
        max_erosion.columns = ['drug_id', 'max_erosion']
        
        high_erosion = max_erosion[max_erosion['max_erosion'] >= threshold]
        
        # Merge with drug info
        if 'brand_name' in df.columns:
            drug_info = df.groupby('drug_id').first()[['brand_name', 'country', 'therapeutic_area']].reset_index()
            high_erosion = high_erosion.merge(drug_info, on='drug_id', how='left')
        
        self.insights.append({
            'type': 'high_erosion_drugs',
            'count': len(high_erosion),
            'threshold': threshold,
            'drugs': high_erosion['drug_id'].tolist() if len(high_erosion) > 0 else []
        })
        
        print(f"\nIdentified {len(high_erosion)} high erosion drugs (threshold: {threshold*100}%)")
        
        return high_erosion
    
    def analyze_erosion_vs_therapeutic_area(self, df: pd.DataFrame,
                                             save_fig: bool = True) -> Dict:
        """
        Analyze relationship between erosion and therapeutic areas.
        
        Args:
            df: DataFrame with erosion and therapeutic area data
            save_fig: Whether to save the figure
            
        Returns:
            Dictionary of insights
        """
        if 'therapeutic_area' not in df.columns or 'erosion' not in df.columns:
            print("Required columns not found")
            return {}
        
        # Calculate mean erosion by therapeutic area
        post_entry = df[df['months_postgx'] >= 0]
        erosion_by_ta = post_entry.groupby('therapeutic_area')['erosion'].agg(['mean', 'std', 'count']).reset_index()
        erosion_by_ta.columns = ['therapeutic_area', 'mean_erosion', 'std_erosion', 'n_observations']
        erosion_by_ta = erosion_by_ta.sort_values('mean_erosion', ascending=False)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        colors = plt.cm.RdYlGn_r(erosion_by_ta['mean_erosion'])
        axes[0].barh(erosion_by_ta['therapeutic_area'], erosion_by_ta['mean_erosion'] * 100, 
                    color=colors, edgecolor='black')
        axes[0].set_xlabel('Mean Erosion (%)', fontsize=12)
        axes[0].set_ylabel('Therapeutic Area', fontsize=12)
        axes[0].set_title('Erosion by Therapeutic Area', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add error bars
        axes[0].errorbar(erosion_by_ta['mean_erosion'] * 100, 
                        range(len(erosion_by_ta)),
                        xerr=erosion_by_ta['std_erosion'] * 100,
                        fmt='none', color='black', capsize=3)
        
        # Box plot over time
        post_entry_sample = post_entry.sample(min(len(post_entry), self.sample_size))
        sns.boxplot(data=post_entry_sample, x='therapeutic_area', y='erosion', ax=axes[1])
        axes[1].set_xlabel('Therapeutic Area', fontsize=12)
        axes[1].set_ylabel('Erosion', fontsize=12)
        axes[1].set_title('Erosion Distribution by Therapeutic Area', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}erosion_vs_therapeutic_area.png', 
                       dpi=300, bbox_inches='tight')
        
        # Generate insights
        highest_erosion_ta = erosion_by_ta.iloc[0]['therapeutic_area']
        lowest_erosion_ta = erosion_by_ta.iloc[-1]['therapeutic_area']
        
        insight = {
            'analysis': 'erosion_vs_therapeutic_area',
            'highest_erosion_area': highest_erosion_ta,
            'highest_erosion_rate': erosion_by_ta.iloc[0]['mean_erosion'],
            'lowest_erosion_area': lowest_erosion_ta,
            'lowest_erosion_rate': erosion_by_ta.iloc[-1]['mean_erosion'],
            'data': erosion_by_ta.to_dict('records')
        }
        
        self.insights.append(insight)
        
        print(f"\nTherapeutic Area Analysis:")
        print(f"  Highest erosion: {highest_erosion_ta} ({insight['highest_erosion_rate']*100:.1f}%)")
        print(f"  Lowest erosion: {lowest_erosion_ta} ({insight['lowest_erosion_rate']*100:.1f}%)")
        
        return insight
    
    def analyze_biological_vs_small_molecule(self, df: pd.DataFrame,
                                              save_fig: bool = True) -> Dict:
        """
        Compare erosion patterns between biological and small molecule drugs.
        
        Args:
            df: DataFrame with drug type and erosion data
            save_fig: Whether to save the figure
            
        Returns:
            Dictionary of insights
        """
        required_cols = ['biological', 'small_molecule', 'erosion']
        if not all(col in df.columns for col in required_cols):
            print("Required columns not found")
            return {}
        
        post_entry = df[df['months_postgx'] >= 0]
        
        # Calculate erosion by drug type
        bio_erosion = post_entry[post_entry['biological'] == 1]['erosion'].mean()
        sm_erosion = post_entry[post_entry['small_molecule'] == 1]['erosion'].mean()
        
        # Time-based analysis
        bio_time = post_entry[post_entry['biological'] == 1].groupby('months_postgx')['erosion'].mean()
        sm_time = post_entry[post_entry['small_molecule'] == 1].groupby('months_postgx')['erosion'].mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar comparison
        x = ['Biological', 'Small Molecule']
        values = [bio_erosion * 100, sm_erosion * 100]
        colors = ['steelblue', 'coral']
        axes[0].bar(x, values, color=colors, edgecolor='black')
        axes[0].set_ylabel('Mean Erosion (%)', fontsize=12)
        axes[0].set_title('Erosion: Biological vs Small Molecule', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(values):
            axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12)
        
        # Time trend comparison
        axes[1].plot(bio_time.index, bio_time.values * 100, 'b-o', 
                    label='Biological', linewidth=2, markersize=4)
        axes[1].plot(sm_time.index, sm_time.values * 100, 'r-s', 
                    label='Small Molecule', linewidth=2, markersize=4)
        axes[1].set_xlabel('Months Post-Generic Entry', fontsize=12)
        axes[1].set_ylabel('Mean Erosion (%)', fontsize=12)
        axes[1].set_title('Erosion Over Time by Drug Type', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}biological_vs_small_molecule.png', 
                       dpi=300, bbox_inches='tight')
        
        insight = {
            'analysis': 'biological_vs_small_molecule',
            'biological_mean_erosion': bio_erosion,
            'small_molecule_mean_erosion': sm_erosion,
            'difference': sm_erosion - bio_erosion,
            'interpretation': 'Small molecules tend to have higher erosion' if sm_erosion > bio_erosion 
                            else 'Biologicals tend to have higher erosion'
        }
        
        self.insights.append(insight)
        
        print(f"\nDrug Type Analysis:")
        print(f"  Biological erosion: {bio_erosion*100:.1f}%")
        print(f"  Small molecule erosion: {sm_erosion*100:.1f}%")
        print(f"  Interpretation: {insight['interpretation']}")
        
        return insight
    
    def analyze_hospital_rate_impact(self, df: pd.DataFrame,
                                      save_fig: bool = True) -> Dict:
        """
        Analyze relationship between hospital rate and erosion.
        
        Args:
            df: DataFrame with hospital rate and erosion data
            save_fig: Whether to save the figure
            
        Returns:
            Dictionary of insights
        """
        if 'hospital_rate' not in df.columns or 'erosion' not in df.columns:
            print("Required columns not found")
            return {}
        
        post_entry = df[df['months_postgx'] >= 0]
        
        # Create hospital rate buckets
        post_entry = post_entry.copy()
        post_entry['hospital_rate_bucket'] = pd.cut(
            post_entry['hospital_rate'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Very High (75-100%)']
        )
        
        # Calculate erosion by hospital rate bucket
        erosion_by_hr = post_entry.groupby('hospital_rate_bucket', observed=True)['erosion'].mean()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot with regression
        axes[0].scatter(post_entry['hospital_rate'], post_entry['erosion'], alpha=0.3, s=20)
        
        # Add trend line
        z = np.polyfit(post_entry['hospital_rate'].dropna(), 
                      post_entry['erosion'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        axes[0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
        
        axes[0].set_xlabel('Hospital Rate', fontsize=12)
        axes[0].set_ylabel('Erosion', fontsize=12)
        axes[0].set_title('Hospital Rate vs Erosion', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Bar plot by bucket
        axes[1].bar(range(len(erosion_by_hr)), erosion_by_hr.values * 100, 
                   color='teal', edgecolor='black')
        axes[1].set_xticks(range(len(erosion_by_hr)))
        axes[1].set_xticklabels(erosion_by_hr.index, rotation=45)
        axes[1].set_xlabel('Hospital Rate Category', fontsize=12)
        axes[1].set_ylabel('Mean Erosion (%)', fontsize=12)
        axes[1].set_title('Erosion by Hospital Rate Category', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}hospital_rate_impact.png', 
                       dpi=300, bbox_inches='tight')
        
        correlation = post_entry['hospital_rate'].corr(post_entry['erosion'])
        
        insight = {
            'analysis': 'hospital_rate_impact',
            'correlation': correlation,
            'trend_slope': z[0],
            'erosion_by_bucket': erosion_by_hr.to_dict(),
            'interpretation': 'Higher hospital rates associated with lower erosion' if correlation < 0
                            else 'Higher hospital rates associated with higher erosion'
        }
        
        self.insights.append(insight)
        
        print(f"\nHospital Rate Impact:")
        print(f"  Correlation with erosion: {correlation:.3f}")
        print(f"  Interpretation: {insight['interpretation']}")
        
        return insight
    
    def analyze_generic_entrants_effect(self, df: pd.DataFrame,
                                         save_fig: bool = True) -> Dict:
        """
        Analyze the effect of number of generic entrants on erosion.
        
        Args:
            df: DataFrame with n_gxs and erosion data
            save_fig: Whether to save the figure
            
        Returns:
            Dictionary of insights
        """
        if 'n_gxs' not in df.columns or 'erosion' not in df.columns:
            print("Required columns not found")
            return {}
        
        post_entry = df[df['months_postgx'] >= 0]
        
        # Erosion by number of generic entrants
        erosion_by_ngxs = post_entry.groupby('n_gxs')['erosion'].agg(['mean', 'std', 'count'])
        erosion_by_ngxs = erosion_by_ngxs[erosion_by_ngxs['count'] >= 10]  # Filter low counts
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Line plot
        axes[0].plot(erosion_by_ngxs.index, erosion_by_ngxs['mean'] * 100, 'b-o', 
                    linewidth=2, markersize=8)
        axes[0].fill_between(erosion_by_ngxs.index, 
                            (erosion_by_ngxs['mean'] - erosion_by_ngxs['std']) * 100,
                            (erosion_by_ngxs['mean'] + erosion_by_ngxs['std']) * 100,
                            alpha=0.3)
        axes[0].set_xlabel('Number of Generic Entrants', fontsize=12)
        axes[0].set_ylabel('Mean Erosion (%)', fontsize=12)
        axes[0].set_title('Erosion vs Number of Generic Entrants', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Heat map: n_gxs vs months_postgx
        pivot = post_entry.pivot_table(values='erosion', index='n_gxs', 
                                       columns='months_postgx', aggfunc='mean')
        sns.heatmap(pivot * 100, cmap='RdYlGn_r', ax=axes[1], 
                   cbar_kws={'label': 'Erosion (%)'})
        axes[1].set_xlabel('Months Post-Generic Entry', fontsize=12)
        axes[1].set_ylabel('Number of Generic Entrants', fontsize=12)
        axes[1].set_title('Erosion Heatmap: Generics vs Time', fontsize=14)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}generic_entrants_effect.png', 
                       dpi=300, bbox_inches='tight')
        
        correlation = post_entry['n_gxs'].corr(post_entry['erosion'])
        
        insight = {
            'analysis': 'generic_entrants_effect',
            'correlation': correlation,
            'max_erosion_at_ngxs': erosion_by_ngxs['mean'].idxmax(),
            'erosion_by_ngxs': erosion_by_ngxs['mean'].to_dict(),
            'interpretation': 'More generic entrants associated with higher erosion'
        }
        
        self.insights.append(insight)
        
        print(f"\nGeneric Entrants Effect:")
        print(f"  Correlation with erosion: {correlation:.3f}")
        print(f"  Maximum erosion at {erosion_by_ngxs['mean'].idxmax()} generics")
        
        return insight
    
    def analyze_bucket1_drugs(self, df: pd.DataFrame,
                               save_fig: bool = True) -> Dict:
        """
        Deep analysis of Bucket 1 (low erosion) drugs.
        
        Args:
            df: DataFrame with erosion bucket assignments
            save_fig: Whether to save the figure
            
        Returns:
            Dictionary of insights
        """
        if 'erosion_bucket' not in df.columns:
            print("Erosion bucket column not found")
            return {}
        
        bucket1 = df[df['erosion_bucket'] == 'Bucket_1']
        
        if len(bucket1) == 0:
            print("No Bucket 1 drugs found")
            return {}
        
        # Characteristics of Bucket 1 drugs
        characteristics = {}
        
        if 'therapeutic_area' in bucket1.columns:
            ta_dist = bucket1.groupby('therapeutic_area')['drug_id'].nunique()
            characteristics['therapeutic_areas'] = ta_dist.to_dict()
        
        if 'biological' in bucket1.columns:
            bio_pct = bucket1['biological'].mean() * 100
            characteristics['biological_percentage'] = bio_pct
        
        if 'hospital_rate' in bucket1.columns:
            avg_hr = bucket1['hospital_rate'].mean()
            characteristics['average_hospital_rate'] = avg_hr
        
        # Volume trend for Bucket 1 drugs
        bucket1_trend = bucket1.groupby('months_postgx')['volume'].mean()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Volume trend
        axes[0, 0].plot(bucket1_trend.index, bucket1_trend.values, 'g-o', linewidth=2)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Generic Entry')
        axes[0, 0].set_xlabel('Months Post-Generic Entry', fontsize=12)
        axes[0, 0].set_ylabel('Average Volume', fontsize=12)
        axes[0, 0].set_title('Bucket 1 Drugs: Volume Trend', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Therapeutic area distribution
        if 'therapeutic_areas' in characteristics:
            ta_data = pd.Series(characteristics['therapeutic_areas'])
            axes[0, 1].barh(ta_data.index, ta_data.values, color='steelblue', edgecolor='black')
            axes[0, 1].set_xlabel('Number of Drugs', fontsize=12)
            axes[0, 1].set_ylabel('Therapeutic Area', fontsize=12)
            axes[0, 1].set_title('Bucket 1: Therapeutic Area Distribution', fontsize=14)
        
        # Hospital rate distribution
        if 'hospital_rate' in bucket1.columns:
            axes[1, 0].hist(bucket1['hospital_rate'].dropna(), bins=20, 
                          color='teal', edgecolor='black', alpha=0.7)
            axes[1, 0].axvline(characteristics.get('average_hospital_rate', 0.5), 
                             color='red', linestyle='--', label='Mean')
            axes[1, 0].set_xlabel('Hospital Rate', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Bucket 1: Hospital Rate Distribution', fontsize=14)
            axes[1, 0].legend()
        
        # Erosion over time comparison
        if 'erosion' in df.columns:
            for bucket in ['Bucket_1', 'Bucket_3', 'Bucket_5']:
                bucket_data = df[df['erosion_bucket'] == bucket]
                if len(bucket_data) > 0:
                    erosion_trend = bucket_data.groupby('months_postgx')['erosion'].mean()
                    axes[1, 1].plot(erosion_trend.index, erosion_trend.values * 100, 
                                  label=bucket, linewidth=2)
            axes[1, 1].set_xlabel('Months Post-Generic Entry', fontsize=12)
            axes[1, 1].set_ylabel('Mean Erosion (%)', fontsize=12)
            axes[1, 1].set_title('Erosion Comparison Across Buckets', fontsize=14)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}bucket1_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        insight = {
            'analysis': 'bucket1_drugs',
            'n_drugs': bucket1['drug_id'].nunique(),
            'characteristics': characteristics,
            'key_finding': 'Bucket 1 drugs show resilience to generic competition'
        }
        
        self.insights.append(insight)
        
        print(f"\nBucket 1 Analysis:")
        print(f"  Number of drugs: {insight['n_drugs']}")
        for key, value in characteristics.items():
            print(f"  {key}: {value}")
        
        return insight
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive text summary of all insights.
        
        Args:
            df: Full analyzed DataFrame
            
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 60)
        report.append("PHARMACEUTICAL DRUG SALES FORECASTING - INSIGHTS REPORT")
        report.append("=" * 60)
        
        # Dataset overview
        report.append("\n1. DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"   Total records: {len(df)}")
        report.append(f"   Unique drugs: {df['drug_id'].nunique()}")
        if 'country' in df.columns:
            report.append(f"   Countries: {df['country'].nunique()}")
        if 'therapeutic_area' in df.columns:
            report.append(f"   Therapeutic areas: {df['therapeutic_area'].nunique()}")
        
        # Key findings
        report.append("\n2. KEY FINDINGS")
        report.append("-" * 40)
        
        for insight in self.insights:
            if insight.get('type') == 'high_erosion_drugs':
                report.append(f"   • {insight['count']} drugs show high erosion (>{insight['threshold']*100}%)")
            
            if insight.get('analysis') == 'erosion_vs_therapeutic_area':
                report.append(f"   • Highest erosion in: {insight['highest_erosion_area']} ({insight['highest_erosion_rate']*100:.1f}%)")
                report.append(f"   • Lowest erosion in: {insight['lowest_erosion_area']} ({insight['lowest_erosion_rate']*100:.1f}%)")
            
            if insight.get('analysis') == 'biological_vs_small_molecule':
                report.append(f"   • {insight['interpretation']}")
            
            if insight.get('analysis') == 'hospital_rate_impact':
                report.append(f"   • Hospital rate correlation: {insight['correlation']:.3f}")
            
            if insight.get('analysis') == 'generic_entrants_effect':
                report.append(f"   • Max erosion at {insight['max_erosion_at_ngxs']} generic entrants")
        
        # Recommendations
        report.append("\n3. BUSINESS RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("   • Focus on high-erosion therapeutic areas for generic defense strategies")
        report.append("   • Prioritize biologicals for longer patent protection value")
        report.append("   • Monitor hospital channel dynamics for erosion prediction")
        report.append("   • Prepare for accelerated erosion with multiple generic entrants")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        with open(f'{self.output_path}insights_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text
    
    def run_full_insights_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete insights analysis.
        
        Args:
            df: Full analyzed DataFrame
            
        Returns:
            Dictionary of all insights
        """
        print("\n" + "="*50)
        print("GENERATING BUSINESS INSIGHTS")
        print("="*50 + "\n")
        
        # Run all analyses
        self.identify_high_erosion_drugs(df)
        self.analyze_erosion_vs_therapeutic_area(df)
        self.analyze_biological_vs_small_molecule(df)
        self.analyze_hospital_rate_impact(df)
        self.analyze_generic_entrants_effect(df)
        self.analyze_bucket1_drugs(df)
        
        # Generate summary
        self.generate_summary_report(df)
        
        # Close all figures to free memory
        plt.close('all')
        
        print(f"\nAll insights saved to: {self.output_path}")
        
        return {'insights': self.insights}
