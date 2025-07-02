
# A Model for Bank Loan Approval Analysis and Predictions
# By: Sandra Ruiz Date: June 29, 2025

import pandas as pd

# Load the first CSV
application_df = pd.read_csv('RuizLoans/Data/application_data.csv')
print("Application Data Preview (First 5 Rows):")
print(application_df.head(5))

# Display row and column count
rows, columns = application_df.shape
print(f"\nThe dataset contains {rows} rows and {columns} columns.")

# Save the first 5000 records to a new CSV
application_sample = application_df.head(5000)
application_sample.to_csv('RuizLoans/Data/application_data_sample.csv', index=False)
print("\n✅ Saved first 5000 records of application_data.csv to application_data_sample.csv")


# Load the second CSV
previous_df = pd.read_csv('RuizLoans/Data/previous_application.csv')
print("\nPrevious Application Data Preview (First 5 Rows):")
print(previous_df.head(5))

# Display row and column count
rows, columns = previous_df.shape
print(f"\nThe previous application dataset contains {rows} rows and {columns} columns.")

# Save the first 5000 records to a new CSV
previous_sample = previous_df.head(5000)
previous_sample.to_csv('RuizLoans/Data/previous_application_sample.csv', index=False)
print("\n✅ Saved first 5000 records of previous_application.csv to previous_application_sample.csv")








