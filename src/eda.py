"""
Exploratory Data Analysis (EDA) Module

This module provides visualization and analysis tools for pharmaceutical
drug sales data exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ExploratoryAnalysis:
    """
    Handles exploratory data analysis and visualization for pharmaceutical
    drug sales forecasting.
    """
    
    def __init__(self, output_path: str = 'outputs/'):
        """
        Initialize the EDA class.
        
        Args:
            output_path: Path to save visualization outputs
        """
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
    def volume_trends_by_country(self, df: pd.DataFrame, 
                                  save_fig: bool = True) -> plt.Figure:
        """
        Visualize volume trends pre- and post-generic entry by country.
        
        Args:
            df: DataFrame containing volume data
            save_fig: Whether to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        countries = df['country'].unique()
        
        for country in countries:
            country_data = df[df['country'] == country]
            avg_volume = country_data.groupby('months_postgx')['volume'].mean()
            ax.plot(avg_volume.index, avg_volume.values, label=country, linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', label='Generic Entry', linewidth=2)
        ax.set_xlabel('Months Post-Generic Entry', fontsize=12)
        ax.set_ylabel('Average Volume', fontsize=12)
        ax.set_title('Volume Trends by Country: Pre- and Post-Generic Entry', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(f'{self.output_path}volume_trends_by_country.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def volume_trends_by_brand(self, df: pd.DataFrame,
                                top_n: int = 10,
                                save_fig: bool = True) -> plt.Figure:
        """
        Visualize volume trends for top brands.
        
        Args:
            df: DataFrame containing volume data
            top_n: Number of top brands to display
            save_fig: Whether to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Get top brands by total volume
        top_brands = df.groupby('brand_name')['volume'].sum().nlargest(top_n).index
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for brand in top_brands:
            brand_data = df[df['brand_name'] == brand]
            brand_vol = brand_data.groupby('months_postgx')['volume'].mean()
            ax.plot(brand_vol.index, brand_vol.values, label=brand, linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', label='Generic Entry', linewidth=2)
        ax.set_xlabel('Months Post-Generic Entry', fontsize=12)
        ax.set_ylabel('Volume', fontsize=12)
        ax.set_title(f'Volume Trends: Top {top_n} Brands', fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(f'{self.output_path}volume_trends_by_brand.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def volume_heatmap_by_month(self, df: pd.DataFrame,
                                 save_fig: bool = True) -> plt.Figure:
        """
        Create a heatmap of volume by months_postgx.
        
        Args:
            df: DataFrame containing volume data
            save_fig: Whether to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Pivot data for heatmap
        pivot_df = df.pivot_table(
            values='volume', 
            index='brand_name', 
            columns='months_postgx', 
            aggfunc='mean'
        )
        
        # Normalize volumes for better visualization
        pivot_normalized = pivot_df.div(pivot_df.max(axis=1), axis=0)
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        sns.heatmap(pivot_normalized, cmap='RdYlGn_r', center=0.5,
                   ax=ax, cbar_kws={'label': 'Normalized Volume'})
        
        ax.set_xlabel('Months Post-Generic Entry', fontsize=12)
        ax.set_ylabel('Brand Name', fontsize=12)
        ax.set_title('Volume Heatmap by Brand and Time', fontsize=14)
        
        if save_fig:
            plt.savefig(f'{self.output_path}volume_heatmap.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def n_gxs_evolution(self, df: pd.DataFrame,
                        save_fig: bool = True) -> plt.Figure:
        """
        Analyze the evolution of number of generic entrants (n_gxs) over time.
        
        Args:
            df: DataFrame containing generics data
            save_fig: Whether to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Average n_gxs over time
        avg_ngxs = df.groupby('months_postgx')['n_gxs'].mean()
        axes[0].plot(avg_ngxs.index, avg_ngxs.values, 'b-o', linewidth=2, markersize=6)
        axes[0].fill_between(avg_ngxs.index, avg_ngxs.values, alpha=0.3)
        axes[0].set_xlabel('Months Post-Generic Entry', fontsize=12)
        axes[0].set_ylabel('Average Number of Generic Entrants', fontsize=12)
        axes[0].set_title('Evolution of Generic Entrants Over Time', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of n_gxs
        if 'n_gxs' in df.columns:
            post_entry = df[df['months_postgx'] >= 0]
            axes[1].hist(post_entry['n_gxs'], bins=20, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Number of Generic Entrants', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Distribution of Generic Entrants', fontsize=14)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}n_gxs_evolution.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def therapeutic_area_analysis(self, df: pd.DataFrame,
                                   save_fig: bool = True) -> plt.Figure:
        """
        Analyze volume trends by therapeutic area.
        
        Args:
            df: DataFrame containing medicine info
            save_fig: Whether to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Volume by therapeutic area
        if 'therapeutic_area' in df.columns:
            ta_volume = df.groupby('therapeutic_area')['volume'].mean()
            axes[0, 0].barh(ta_volume.index, ta_volume.values, color='steelblue')
            axes[0, 0].set_xlabel('Average Volume', fontsize=12)
            axes[0, 0].set_ylabel('Therapeutic Area', fontsize=12)
            axes[0, 0].set_title('Average Volume by Therapeutic Area', fontsize=14)
        
        # Plot 2: Hospital rate distribution
        if 'hospital_rate' in df.columns:
            axes[0, 1].hist(df['hospital_rate'].dropna(), bins=20, 
                          edgecolor='black', alpha=0.7, color='coral')
            axes[0, 1].set_xlabel('Hospital Rate', fontsize=12)
            axes[0, 1].set_ylabel('Frequency', fontsize=12)
            axes[0, 1].set_title('Distribution of Hospital Rate', fontsize=14)
        
        # Plot 3: Biological vs Small Molecule
        if 'biological' in df.columns and 'small_molecule' in df.columns:
            bio_vol = df.groupby('biological')['volume'].mean()
            sm_vol = df.groupby('small_molecule')['volume'].mean()
            
            x = ['Biological', 'Small Molecule']
            yes_vals = [bio_vol.get(1, 0), sm_vol.get(1, 0)]
            no_vals = [bio_vol.get(0, 0), sm_vol.get(0, 0)]
            
            x_pos = np.arange(len(x))
            width = 0.35
            
            axes[1, 0].bar(x_pos - width/2, yes_vals, width, label='Yes', color='green', alpha=0.7)
            axes[1, 0].bar(x_pos + width/2, no_vals, width, label='No', color='gray', alpha=0.7)
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(x)
            axes[1, 0].set_ylabel('Average Volume', fontsize=12)
            axes[1, 0].set_title('Volume: Biological vs Small Molecule', fontsize=14)
            axes[1, 0].legend()
        
        # Plot 4: Erosion by therapeutic area
        if 'therapeutic_area' in df.columns:
            post_entry = df[df['months_postgx'] >= 0]
            if len(post_entry) > 0:
                ta_erosion = post_entry.groupby(['therapeutic_area', 'months_postgx'])['volume'].mean().reset_index()
                for ta in ta_erosion['therapeutic_area'].unique():
                    ta_data = ta_erosion[ta_erosion['therapeutic_area'] == ta]
                    axes[1, 1].plot(ta_data['months_postgx'], ta_data['volume'], 
                                  label=ta, linewidth=2)
                axes[1, 1].set_xlabel('Months Post-Generic Entry', fontsize=12)
                axes[1, 1].set_ylabel('Average Volume', fontsize=12)
                axes[1, 1].set_title('Post-Entry Volume by Therapeutic Area', fontsize=14)
                axes[1, 1].legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}therapeutic_area_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def erosion_analysis(self, df: pd.DataFrame,
                         save_fig: bool = True) -> plt.Figure:
        """
        Analyze volume erosion patterns post-generic entry.
        
        Args:
            df: DataFrame containing volume and time data
            save_fig: Whether to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Calculate erosion metrics per drug
        pre_entry = df[df['months_postgx'] < 0]
        post_entry = df[df['months_postgx'] >= 0]
        
        pre_avg = pre_entry.groupby('drug_id')['volume'].mean()
        
        erosion_data = []
        for drug_id in pre_avg.index:
            drug_post = post_entry[post_entry['drug_id'] == drug_id]
            if len(drug_post) > 0:
                for month in drug_post['months_postgx'].unique():
                    month_vol = drug_post[drug_post['months_postgx'] == month]['volume'].mean()
                    erosion = 1 - (month_vol / pre_avg[drug_id]) if pre_avg[drug_id] > 0 else 0
                    erosion_data.append({
                        'drug_id': drug_id,
                        'months_postgx': month,
                        'erosion': max(0, min(1, erosion))  # Clip between 0 and 1
                    })
        
        erosion_df = pd.DataFrame(erosion_data)
        
        # Plot 1: Average erosion over time
        avg_erosion = erosion_df.groupby('months_postgx')['erosion'].mean()
        axes[0, 0].plot(avg_erosion.index, avg_erosion.values * 100, 'r-o', linewidth=2)
        axes[0, 0].fill_between(avg_erosion.index, avg_erosion.values * 100, alpha=0.3, color='red')
        axes[0, 0].set_xlabel('Months Post-Generic Entry', fontsize=12)
        axes[0, 0].set_ylabel('Average Erosion (%)', fontsize=12)
        axes[0, 0].set_title('Volume Erosion Over Time', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of erosion at month 12
        month_12 = erosion_df[erosion_df['months_postgx'] == 12] if 12 in erosion_df['months_postgx'].values else erosion_df
        axes[0, 1].hist(month_12['erosion'] * 100, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Erosion (%)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution of Erosion (Month 12)', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Erosion by drug (box plot for each time period)
        time_periods = [6, 12, 18, 23]
        erosion_by_period = []
        labels = []
        for period in time_periods:
            period_data = erosion_df[erosion_df['months_postgx'] == period]['erosion'].values * 100
            if len(period_data) > 0:
                erosion_by_period.append(period_data)
                labels.append(f'Month {period}')
        
        if erosion_by_period:
            axes[1, 0].boxplot(erosion_by_period, labels=labels)
            axes[1, 0].set_ylabel('Erosion (%)', fontsize=12)
            axes[1, 0].set_title('Erosion Distribution by Time Period', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative erosion
        final_erosion = erosion_df.groupby('drug_id')['erosion'].max()
        axes[1, 1].hist(final_erosion * 100, bins=20, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Final Erosion (%)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Distribution of Maximum Erosion', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_path}erosion_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing summary statistics
        """
        stats = {
            'total_records': len(df),
            'unique_drugs': df['drug_id'].nunique() if 'drug_id' in df.columns else None,
            'unique_countries': df['country'].nunique() if 'country' in df.columns else None,
            'date_range': {
                'min_month': df['months_postgx'].min() if 'months_postgx' in df.columns else None,
                'max_month': df['months_postgx'].max() if 'months_postgx' in df.columns else None
            },
            'volume_stats': {
                'mean': df['volume'].mean() if 'volume' in df.columns else None,
                'std': df['volume'].std() if 'volume' in df.columns else None,
                'min': df['volume'].min() if 'volume' in df.columns else None,
                'max': df['volume'].max() if 'volume' in df.columns else None
            }
        }
        
        print("\n=== Summary Statistics ===")
        print(f"Total Records: {stats['total_records']}")
        print(f"Unique Drugs: {stats['unique_drugs']}")
        print(f"Unique Countries: {stats['unique_countries']}")
        print(f"Time Range: {stats['date_range']['min_month']} to {stats['date_range']['max_month']} months")
        print(f"Volume Mean: {stats['volume_stats']['mean']:.2f}")
        print(f"Volume Std: {stats['volume_stats']['std']:.2f}")
        
        return stats
    
    def run_full_eda(self, df: pd.DataFrame) -> Dict:
        """
        Run complete exploratory data analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of statistics and insights
        """
        print("\n" + "="*50)
        print("RUNNING EXPLORATORY DATA ANALYSIS")
        print("="*50 + "\n")
        
        # Generate summary statistics
        stats = self.summary_statistics(df)
        
        # Generate all visualizations
        print("\nGenerating visualizations...")
        
        self.volume_trends_by_country(df)
        print("✓ Volume trends by country")
        
        self.volume_trends_by_brand(df)
        print("✓ Volume trends by brand")
        
        if 'n_gxs' in df.columns:
            self.n_gxs_evolution(df)
            print("✓ Generic entrants evolution")
        
        if 'therapeutic_area' in df.columns:
            self.therapeutic_area_analysis(df)
            print("✓ Therapeutic area analysis")
        
        self.erosion_analysis(df)
        print("✓ Erosion analysis")
        
        # Close all figures to free memory
        plt.close('all')
        
        print(f"\nAll visualizations saved to: {self.output_path}")
        
        return stats
