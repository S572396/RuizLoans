# Data Statistics Previous_application csv
import pandas as pd

# Load the previous application sample file
file_path = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\previous_application_sample.csv'
df = pd.read_csv(file_path)

# Output file path (updated)
output_file = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\prev_app_answer.txt'

# List to store lines for saving
output_lines = []

def log_and_print(message):
    print(message)
    output_lines.append(message)

# Filter for Cash Loans
cash_loans_df = df[df['NAME_CONTRACT_TYPE'] == 'Cash loans']
cash_status_counts = cash_loans_df['NAME_CONTRACT_STATUS'].value_counts()

# Filter for Revolving Loans
revolving_loans_df = df[df['NAME_CONTRACT_TYPE'] == 'Revolving loans']
revolving_status_counts = revolving_loans_df['NAME_CONTRACT_STATUS'].value_counts()

# Print and save results
log_and_print("\nCash Loans Approval Status Counts:")
log_and_print(str(cash_status_counts))

log_and_print("\nRevolving Loans Approval Status Counts:")
log_and_print(str(revolving_status_counts))

# Write results to file
with open(output_file, 'w') as file:
    for line in output_lines:
        file.write(line + '\n')

print(f"\n✅ Results successfully saved to: {output_file}")
