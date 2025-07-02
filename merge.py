import pandas as pd

# Load files
app_df = pd.read_csv(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\application_data_sample.csv')
prev_df = pd.read_csv(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\previous_application_sample.csv')

# Merge previous loans with new application data on SK_ID_CURR
merged_df = pd.merge(prev_df, app_df, on='SK_ID_CURR', how='left')

# Save merged dataframe to new CSV file named 'data_merged.csv'
output_file = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\data_merged.csv'
merged_df.to_csv(output_file, index=False)

print(f"Merged data saved successfully to: {output_file}")
print(merged_df.head())
