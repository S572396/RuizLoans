# Data Statistics application csv file

import pandas as pd

# Load the sample CSV
file_path = r"C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\application_data_sample.csv"
df = pd.read_csv(file_path)

# Prepare output file
output_file = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\application_stat_answers.txt'

# Create a list to store all answer strings
output_lines = []


def log_and_print(message):
    """Print to terminal and store in output list."""
    print(message)
    output_lines.append(message)


# Gender Counts
try:
    gender_counts = df['CODE_GENDER'].value_counts()
    log_and_print("\nGender Counts:")
    log_and_print(str(gender_counts))
except KeyError:
    log_and_print("\nColumn 'CODE_GENDER' not found.")

# Car owned
try:
    car_counts = df['FLAG_OWN_CAR'].value_counts()
    log_and_print("\nCar Ownership Counts:")
    log_and_print(str(car_counts))
except KeyError:
    log_and_print("\nColumn 'FLAG_OWN_CAR' not found.")

# Realty or Home Ownership
try:
    realty_counts = df['FLAG_OWN_REALTY'].value_counts()
    log_and_print("\nRealty Ownership Counts:")
    log_and_print(str(realty_counts))
except KeyError:
    log_and_print("\nColumn 'FLAG_OWN_REALTY' not found.")



# Age Statistics
try:
    # Convert negative days to positive years
    df['AGE_YEARS'] = (df['DAYS_BIRTH'] / -365).round(1)

    max_age = df['AGE_YEARS'].max()
    min_age = df['AGE_YEARS'].min()
    avg_age = df['AGE_YEARS'].mean().round(1)

    log_and_print("\nAge Statistics:")
    log_and_print(f"Oldest Age: {max_age} years")
    log_and_print(f"Youngest Age: {min_age} years")
    log_and_print(f"Average Age: {avg_age} years")

except KeyError:
    log_and_print("\nColumn 'DAYS_BIRTH' not found.")

# Occupations
try:
    # Remove any Nan
    df_clean = df.dropna(subset=['OCCUPATION_TYPE'])
    occupation_counts = df_clean['OCCUPATION_TYPE'].value_counts()
    log_and_print("\nOccupation Types Among Records:")
    log_and_print(str(occupation_counts))

except KeyError:
    log_and_print("\nColumn 'OCCUPATION_TYPE' not found.")





# Cash or Revolving Loan type
try:
    contract_type_counts = df['NAME_CONTRACT_TYPE'].value_counts()
    log_and_print("\nLoan Type Counts:")
    log_and_print(str(contract_type_counts))

    cash_loans = contract_type_counts.get('Cash loans', 0)
    revolving_loans = contract_type_counts.get('Revolving loans', 0)

    log_and_print(f"\nNumber of Cash Loan Applications: {cash_loans}")
    log_and_print(f"Number of Revolving Loan Applications: {revolving_loans}")

except KeyError:
    log_and_print("\nColumn 'NAME_CONTRACT_TYPE' not found.")

# Married status
try:
    marital_status_counts = df['NAME_FAMILY_STATUS'].value_counts()
    log_and_print("\nMarital Status Counts:")
    log_and_print(str(marital_status_counts))

    married_count = marital_status_counts.get('Married', 0)
    log_and_print(f"\nNumber of Married Applicants: {married_count}")

except KeyError:
    log_and_print("\nColumn 'NAME_FAMILY_STATUS' not found.")

# Children
try:
    applicants_with_children = df[df['CNT_CHILDREN'] > 0].shape[0]
    log_and_print(f"\nNumber of Applicants with Children: {applicants_with_children}")
except KeyError:
    log_and_print("\nColumn 'CNT_CHILDREN' not found.")

# Looking at Income
try:
    max_income = df['AMT_INCOME_TOTAL'].max()
    min_income = df['AMT_INCOME_TOTAL'].min()
    avg_income = df['AMT_INCOME_TOTAL'].mean()

    log_and_print(f"\nHighest Income: {max_income}")
    log_and_print(f"Lowest Income: {min_income}")
    log_and_print(f"Average Income: {avg_income:.2f}")

except KeyError:
    log_and_print("\nColumn 'AMT_INCOME_TOTAL' not found in the dataset.")

# Available Credit
try:
    max_credit = df['AMT_CREDIT'].max()
    min_credit = df['AMT_CREDIT'].min()
    avg_credit = df['AMT_CREDIT'].mean()

    log_and_print(f"\nHighest Credit Amount: {max_credit}")
    log_and_print(f"Lowest Credit Amount: {min_credit}")
    log_and_print(f"Average Credit Amount: {avg_credit:.2f}")

except KeyError:
    log_and_print("\nColumn 'AMT_CREDIT' not found in the dataset.")

# --- Save All Results to File ---
with open(output_file, 'w') as file:
    for line in output_lines:
        file.write(line + '\n')

print(f"\n✅ Results successfully saved to: {output_file}")






