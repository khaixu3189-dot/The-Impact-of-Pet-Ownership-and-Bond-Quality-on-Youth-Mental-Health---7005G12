## Data Cleaning
import pandas as pd

# 1. Load raw data
path = "C:/Users/huahua-zz/Desktop/group project/data_unclean.xlsx"
df = pd.read_excel(path)

# 2. Columns to DROP (21 variables unnecessary for RQ1 + RQ2)
drop_cols = [
    'num_bird','num_fish','num_ferret','num_gerbil','num_guinea','num_hamster',
    'num_horse','num_iguana','num_mice','num_rabbit','num_rat','num_snake',
    'num_tarantula','num_turtle','num_gecko','num_snail', 'living_status',
    'gender_aab', 'income', 'sexuality', 'ethnicity', 'employment', 'mh_dx_type',
    'gender_dummy', 'sexuality_dummy', 'ethnicity_dummy'
]
df = df.drop(columns=drop_cols)

# 3. Convert key variables to proper types
# 3.1 pet_owner: convert Yes/No to binary 1/0
df['pet_owner_bin'] = df['pet_owner'].map({'Yes': 1, 'No': 0})

# 3.2 Set categorical variables as category dtype
cat_cols = ['gender', 'mh_dx', 'sh_status', 'pet_owner']
for c in cat_cols:
    df[c] = df[c].astype('category')

# 3.3 Ensure numeric variables are numeric dtype
num_cols = [
    'age','num_pets','pet_diversity','num_cat','num_dog',
    'prs_affective','prs_family','prs_activity','prs_total',
    'hads_anxiety','hads_depression',
    'sh_severity','sh_diversity','sh_time_weeks',
    'sbq_total'
]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# 4. Inspect missing values (especially PRS)
print("Missing counts in numeric variables:")
print(df[num_cols].isna().sum())
print("\nMissing counts in PRS variables:")
print(df[['prs_affective','prs_family','prs_activity','prs_total']].isna().sum())

# 5. Save clean dataset for analysis
df.to_csv("C:/Users/huahua-zz/Desktop/group project/data_clean.csv", index=False)

#RQ1: Does pet ownership and the quality of the pet–owner bond influence mental health of youth?
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/huahua-zz/Desktop/group project/data_clean.csv")

## RQ1: Overall Descriptive Statistics
def descriptive_tables(df, num_vars):
    df = df.copy()
    for v in num_vars:
        df[v] = pd.to_numeric(df[v])
    print("\n# Overall Descriptive Statistics:")
    print(df[num_vars].describe().T)

num_vars = ['num_pets', 'pet_diversity', 'prs_total', 
            'hads_anxiety', 'hads_depression', 'sh_severity', 'sbq_total']
descriptive_tables(df, num_vars)

## RQ1: Grouped Descriptive Statistics
def plot_petowner_boxplot(df, mh_vars):
    for var in mh_vars:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='pet_owner', y=var, data=df, palette='Set2', showmeans=True)
        plt.title(f'{var} by Pet Owner')
        plt.xlabel('Pet Owner')
        plt.ylabel(var)
        plt.tight_layout()
        plt.show()

mh_vars = ['hads_anxiety', 'hads_depression', 'sh_severity', 'sbq_total']
plot_petowner_boxplot(df, mh_vars)

## RQ1: Grouped Descriptive Statistics
def plot_petowner_boxplot(df, mh_vars):
    for var in mh_vars:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='pet_owner', y=var, data=df, palette='Set2', showmeans=True)
        plt.title(f'{var} by Pet Owner')
        plt.xlabel('Pet Owner')
        plt.ylabel(var)
        plt.tight_layout()
        plt.show()

mh_vars = ['hads_anxiety', 'hads_depression', 'sh_severity', 'sbq_total']
plot_petowner_boxplot(df, mh_vars)

## RQ1: pet_owner statistical test
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, levene, norm

def test_pet_owner_effect(df, mh_var):
    # Map pet ownership
    df["pet_owner"] = df["pet_owner_bin"].map({1:"owner", 0:"non_owner"})
    owners = df[df.pet_owner=="owner"][mh_var].dropna()
    nonowners = df[df.pet_owner=="non_owner"][mh_var].dropna()

    # 1. Normality Test
    p1 = shapiro(owners).pvalue
    p2 = shapiro(nonowners).pvalue
    print(f"\n# Variable tested: {mh_var}")
    if p1 > 0.05 and p2 > 0.05:
        # 2. Homogeneity Test of Variance
        levene_p = levene(owners, nonowners).pvalue
        if levene_p > 0.05:
            print("Use standard t-test (equal variance)")
            test_result = ttest_ind(owners, nonowners, equal_var=True)
        else:
            print("Use Welch t-test (unequal variance)")
            test_result = ttest_ind(owners, nonowners, equal_var=False)
    else:
        print("Use Mann–Whitney U test")
        test_result = mannwhitneyu(owners, nonowners, alternative='two-sided')
    print("Test statistics:", test_result)

    # 3. Calculate the Mean
    print("owners mean:", owners.mean())
    print("non-owners mean:", nonowners.mean())

    # 4. Calculate the Median
    print("owners median:", owners.median())
    print("non-owners median:", nonowners.median())

    # 5. Calculate the Effect Size (r = z / sqrt(n))
    res = mannwhitneyu(owners, nonowners, alternative='two-sided')
    U = res.statistic
    p = res.pvalue
    Z = norm.ppf(p/2) * -1
    N = len(owners) + len(nonowners)
    r = abs(Z) / (N ** 0.5)
    print("effect size r =", r)

# Using function
for outcome in ["hads_anxiety", "hads_depression", "sh_severity", "sbq_total"]:
    test_pet_owner_effect(df, outcome)

## RQ1: bond quality statistical test
def test_prstotal_group(df, outcome_col):
    # 1. Group by prs_total
    low_bond = df[df['prs_total'] <= 3][outcome_col].dropna()
    high_bond = df[df['prs_total'] > 3][outcome_col].dropna()

    # 2. Normality Test
    p1 = shapiro(low_bond).pvalue
    p2 = shapiro(high_bond).pvalue
    print(f"\n# Outcome variable: {outcome_col}")
    if p1 > 0.05 and p2 > 0.05:
        # 3. Homogeneity Test of Variance
        levene_p = levene(low_bond, high_bond).pvalue
        if levene_p > 0.05:
            print("Use standard t-test (equal variance)")
            test_result = ttest_ind(low_bond, high_bond, equal_var=True)
        else:
            print("Use Welch t-test (unequal variance)")
            test_result = ttest_ind(low_bond, high_bond, equal_var=False)
    else:
        print("Use Mann–Whitney U test")
        test_result = mannwhitneyu(low_bond, high_bond, alternative='two-sided')
    print("Test statistics:", test_result)

    # 4. Calculate the Mean
    print("low bond mean:", low_bond.mean())
    print("high bond mean:", high_bond.mean())

    # 5. Calculate the Effect Size (r = z / sqrt(n))
    res = mannwhitneyu(low_bond, high_bond, alternative='two-sided')
    U = res.statistic
    p = res.pvalue
    Z = norm.ppf(p/2) * -1
    N = len(low_bond) + len(high_bond)
    r = abs(Z) / (N ** 0.5)
    print("effect size r =", r)

    # 6. Calculate the Median
    print("low bond median:", low_bond.median())
    print("high bond median:", high_bond.median())

# Using function:
for outcome in ["hads_anxiety", "hads_depression", "sh_severity", "sbq_total"]:
    test_prstotal_group(df, outcome)

## RQ1: Heatmap：Pet variables × Mental Health outcomes
import numpy as np
from scipy.stats import spearmanr

data = pd.read_csv('C:/Users/huahua-zz/Desktop/group project/data_clean.csv')

# 1. Variable Selection and Preprocessing
pet_vars = ['pet_owner', 'num_pets', 'pet_diversity', 'prs_total']
mh_vars = ['hads_anxiety', 'hads_depression', 'sh_severity', 'sbq_total']
df = data[pet_vars + mh_vars].copy()
df['pet_owner'] = df['pet_owner'].map({'Yes': 1, 'No': 0}).astype(float)

# 2. Correlation and Significance CaLculation (handling missing values for each pair of variables)
corr_matrix = np.zeros((len(pet_vars), len(mh_vars)))
p_matrix = np.zeros((len(pet_vars), len(mh_vars)))
for i, pet in enumerate(pet_vars):
    for j, mh in enumerate(mh_vars):
        sub_df = df[[pet, mh]].dropna() 
        if sub_df[pet].nunique() < 2 or sub_df[mh].nunique() < 2:
            coef, pval = np.nan, np.nan
        else:
            coef, pval = spearmanr(sub_df[pet], sub_df[mh])
        corr_matrix[i, j] = coef
        p_matrix[i, j] = pval

# 3. Mark the Correlation Coefficient and Significance
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

labels = np.array([f"{corr_matrix[i,j]:.2f}{significance_stars(p_matrix[i,j])}"
                   for i in range(len(pet_vars)) for j in range(len(mh_vars))])
labels = labels.reshape(corr_matrix.shape)

# 4. Visualization
plt.figure(figsize=(9,6))
sns.set(font_scale=1.15, style='white')
cmap = 'coolwarm'
ax = sns.heatmap(
    corr_matrix, 
    annot=labels, 
    fmt='', 
    cmap=cmap,
    vmin=-1, vmax=1, 
    xticklabels=mh_vars, 
    yticklabels=pet_vars,
    cbar_kws={"label": "Spearman correlation", "shrink": 0.8},
    annot_kws={"size": 13, "weight": "normal", "color": "black"}, 
    linewidths=0 
)
plt.xticks(fontsize=13, weight='normal', rotation=0)
plt.yticks(fontsize=13, weight='normal', rotation=0)
plt.title("Correlation Heatmap: Pet variables × Mental health outcomes\n(*significance, P<0.05/*, P<0.01/**, P<0.001/***)", 
          fontsize=15, weight="bold", pad=20)
plt.tight_layout()
plt.show()
