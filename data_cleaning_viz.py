import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
# ─────────────────────────────────────────
# 1. GENERATE RAW DATASET
# ─────────────────────────────────────────
np.random.seed(42)
n = 300
 
raw_data = {
    'age':        np.random.randint(18, 65, n).astype(float),
    'salary':     np.random.normal(55000, 15000, n),
    'department': np.random.choice(['HR', 'Engineering', 'Sales', 'Marketing', None], n),
    'score':      np.random.uniform(0, 100, n),
    'experience': np.random.randint(0, 20, n).astype(float),
    'gender':     np.random.choice(['Male', 'Female', 'male', 'female', None, 'M', 'F'], n),
}
 
df = pd.DataFrame(raw_data)
 
# Inject problems
df.loc[np.random.choice(n, 30, replace=False), 'age']        = np.nan
df.loc[np.random.choice(n, 25, replace=False), 'salary']     = np.nan
df.loc[np.random.choice(n, 10, replace=False), 'salary']     = np.random.uniform(200000, 500000, 10)  # outliers
df.loc[np.random.choice(n, 5,  replace=False), 'age']        = np.random.uniform(100, 150, 5)          # outliers
df = pd.concat([df, df.sample(15)], ignore_index=True)       # duplicates
 
print("=" * 55)
print("  RAW DATASET OVERVIEW")
print("=" * 55)
print(f"  Shape            : {df.shape}")
print(f"  Missing values   :\n{df.isnull().sum().to_string()}")
print(f"  Duplicates       : {df.duplicated().sum()}")
print()
 
# ─────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────
 
# Step 1 – Drop duplicates
df.drop_duplicates(inplace=True)
 
# Step 2 – Standardise gender
gender_map = {'male': 'Male', 'M': 'Male', 'female': 'Female', 'F': 'Female'}
df['gender'] = df['gender'].replace(gender_map)
df['gender'].fillna('Unknown', inplace=True)
 
# Step 3 – Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['salary'].fillna(df['salary'].median(), inplace=True)
df['department'].fillna('Unknown', inplace=True)
 
# Step 4 – Remove outliers (IQR method)
def remove_outliers(data, col):
    Q1, Q3 = data[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]
 
df = remove_outliers(df, 'salary')
df = remove_outliers(df, 'age')
df.reset_index(drop=True, inplace=True)
 
# Step 5 – Feature engineering
df['seniority'] = pd.cut(df['experience'],
                         bins=[-1, 2, 7, 14, 20],
                         labels=['Junior', 'Mid', 'Senior', 'Lead'])
 
print("=" * 55)
print("  CLEANED DATASET OVERVIEW")
print("=" * 55)
print(f"  Shape            : {df.shape}")
print(f"  Missing values   : {df.isnull().sum().sum()}")
print(f"  Duplicates       : {df.duplicated().sum()}")
print()
 
# ─────────────────────────────────────────
# 3. VISUALISATION DASHBOARD
# ─────────────────────────────────────────
 
plt.style.use('seaborn-v0_8-whitegrid')
ACCENT  = '#4F46E5'
ACCENT2 = '#10B981'
BG      = '#F8F9FB'
TEXT    = '#1E1B4B'
 
fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle('Employee Dataset — Cleaning & Insights Dashboard',
             fontsize=20, fontweight='bold', color=TEXT, y=0.98)
 
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
 
# ── Plot 1: Salary Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['salary'], bins=30, color=ACCENT, edgecolor='white', alpha=0.85)
ax1.axvline(df['salary'].mean(), color='crimson', linestyle='--', linewidth=1.5, label=f"Mean: ${df['salary'].mean():,.0f}")
ax1.set_title('Salary Distribution', fontweight='bold', color=TEXT)
ax1.set_xlabel('Salary ($)')
ax1.legend(fontsize=8)
 
# ── Plot 2: Avg Salary by Department
ax2 = fig.add_subplot(gs[0, 1])
dept_salary = df.groupby('department')['salary'].mean().sort_values()
bars = ax2.barh(dept_salary.index, dept_salary.values, color=ACCENT, alpha=0.85)
ax2.bar_label(bars, fmt='$%.0f', padding=4, fontsize=8)
ax2.set_title('Avg Salary by Department', fontweight='bold', color=TEXT)
ax2.set_xlabel('Avg Salary ($)')
 
# ── Plot 3: Age Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df['age'], bins=25, color=ACCENT2, edgecolor='white', alpha=0.85)
ax3.axvline(df['age'].mean(), color='crimson', linestyle='--', linewidth=1.5, label=f"Mean: {df['age'].mean():.1f}")
ax3.set_title('Age Distribution', fontweight='bold', color=TEXT)
ax3.set_xlabel('Age')
ax3.legend(fontsize=8)
 
# ── Plot 4: Score by Seniority (Box)
ax4 = fig.add_subplot(gs[1, 0])
palette = {'Junior': '#818CF8', 'Mid': '#4F46E5', 'Senior': '#3730A3', 'Lead': '#1E1B4B'}
sns.boxplot(data=df, x='seniority', y='score', ax=ax4,
            order=['Junior', 'Mid', 'Senior', 'Lead'],
            palette=palette)
ax4.set_title('Score by Seniority Level', fontweight='bold', color=TEXT)
ax4.set_xlabel('Seniority')
ax4.set_ylabel('Score')
 
# ── Plot 5: Salary vs Experience (Scatter)
ax5 = fig.add_subplot(gs[1, 1])
scatter = ax5.scatter(df['experience'], df['salary'],
                      c=df['score'], cmap='viridis', alpha=0.6, s=30)
plt.colorbar(scatter, ax=ax5, label='Score')
m, b = np.polyfit(df['experience'], df['salary'], 1)
xr = np.linspace(df['experience'].min(), df['experience'].max(), 100)
ax5.plot(xr, m * xr + b, color='crimson', linewidth=1.5, label='Trend')
ax5.set_title('Salary vs Experience', fontweight='bold', color=TEXT)
ax5.set_xlabel('Experience (yrs)')
ax5.set_ylabel('Salary ($)')
ax5.legend(fontsize=8)
 
# ── Plot 6: Gender Breakdown (Pie)
ax6 = fig.add_subplot(gs[1, 2])
gender_counts = df['gender'].value_counts()
colors = [ACCENT, ACCENT2, '#F59E0B', '#EF4444']
wedges, texts, autotexts = ax6.pie(
    gender_counts, labels=gender_counts.index,
    autopct='%1.1f%%', startangle=140,
    colors=colors[:len(gender_counts)],
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
for t in autotexts: t.set_fontsize(8)
ax6.set_title('Gender Distribution', fontweight='bold', color=TEXT)
 
# ── Plot 7: Correlation Heatmap
ax7 = fig.add_subplot(gs[2, 0:2])
corr = df[['age', 'salary', 'score', 'experience']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax7, linewidths=0.5,
            annot_kws={'size': 10})
ax7.set_title('Correlation Heatmap', fontweight='bold', color=TEXT)
 
# ── Plot 8: Headcount by Department
ax8 = fig.add_subplot(gs[2, 2])
dept_count = df['department'].value_counts()
ax8.bar(dept_count.index, dept_count.values, color=ACCENT, alpha=0.85, edgecolor='white')
ax8.set_title('Headcount by Department', fontweight='bold', color=TEXT)
ax8.set_xlabel('Department')
ax8.set_ylabel('Count')
plt.setp(ax8.get_xticklabels(), rotation=30, ha='right', fontsize=8)
 
plt.savefig('/mnt/user-data/outputs/dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Dashboard saved → dashboard.png")
plt.show()
 
# ─────────────────────────────────────────
# 4. SUMMARY STATS
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY STATISTICS (CLEANED DATA)")
print("=" * 55)
print(df[['age', 'salary', 'score', 'experience']].describe().round(2).to_string())
print()
print("  Seniority breakdown:")
print(df['seniority'].value_counts().to_string())
 
