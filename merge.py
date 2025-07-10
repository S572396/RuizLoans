import pandas as pd

# Load files
app_df = pd.read_csv(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\application_data_sample.csv')
prev_df = pd.read_csv(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\previous_application_sample.csv')

# Merge previous loans with new application data on SK_ID_CURR
merged_df = pd.merge(prev_df, app_df, on='SK_ID_CURR', how='left')

# Partial match keywords to remove columns
partial_keywords = [
    'AMT_ANNUITY', 'AMT_GOODS', 'REGION', 'ENTRANCES', 'FLOORS', 'WALLSTREET', 'LIVINGAPARTMENTS','EMERGENCY'
    'ELEVATORS', 'NONLIVING', 'SOCIAL', 'BASEMENT', 'YEARS_BUILD', 'FLAG_DOCUMENT','AMT_REQ_CREDIT_BUREAU'
]

# Exact column names to drop
exact_columns = [
    'NAME_CASH_LOAN_PURPOSE', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'APARTMENTS_AVG',
    'YEARS_BEGINEXPLUATATION_AVG', 'LIVINGAREA_AVG', 'APARTMENTS_MODE',
    'YEARS_BEGINEXPLUATATION_MODE', 'LIVINGAREA_MODE', 'APARTMENTS_MEDI',
    'YEARS_BEGINEXPLUATATION_MEDI', 'LIVINGAREA_MEDI', 'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'DAYS_LAST_PHONE_CHANGE'
]

# Drop partial match columns
for kw in partial_keywords:
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains(kw, case=False, regex=False)]

# Drop exact match columns
merged_df = merged_df.drop(columns=[col for col in exact_columns if col in merged_df.columns])

# Save merged dataframe
output_file = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\data_merged.csv'
merged_df.to_csv(output_file, index=False)

print(f"Merged data saved successfully to: {output_file}")
print("Preview of merged data:")
print(merged_df.head())


