# A Model for Bank Loan Approval Analysis and Predictions
# By: Sandra Ruiz Date: June 29, 2025

import pandas as pd

# Open the output file with UTF-8 encoding
with open('RuizLoans/Data/describe.txt', 'w', encoding='utf-8') as file:

    # Load the first CSV
    application_df = pd.read_csv('RuizLoans/Data/application_data.csv')
    preview = "Application Data Preview (First 5 Rows):\n"
    preview += str(application_df.head(5)) + "\n"
    print(preview)
    file.write(preview + "\n")

    # Display row and column count
    rows, columns = application_df.shape
    shape_info = f"The dataset contains {rows} rows and {columns} columns.\n"
    print(shape_info)
    file.write(shape_info + "\n")

    # Display dataset summary statistics
    describe_info = "Application Data Summary Statistics:\n"
    describe_info += str(application_df.describe(include='all')) + "\n"
    print(describe_info)
    file.write(describe_info + "\n")

    # Save the first 5000 records to a new CSV
    application_sample = application_df.head(5000)
    application_sample.to_csv('RuizLoans/Data/application_data_sample.csv', index=False)
    save_message = "Saved first 5000 records of application_data.csv to application_data_sample.csv\n"
    print(save_message)
    file.write(save_message + "\n")

    # Load the second CSV
    previous_df = pd.read_csv('RuizLoans/Data/previous_application.csv')
    preview_prev = "Previous Application Data Preview (First 5 Rows):\n"
    preview_prev += str(previous_df.head(5)) + "\n"
    print(preview_prev)
    file.write(preview_prev + "\n")

    # Display row and column count
    rows, columns = previous_df.shape
    shape_info_prev = f"The previous application dataset contains {rows} rows and {columns} columns.\n"
    print(shape_info_prev)
    file.write(shape_info_prev + "\n")

    # Display dataset summary statistics
    describe_info_prev = "Previous Application Data Summary Statistics:\n"
    describe_info_prev += str(previous_df.describe(include='all')) + "\n"
    print(describe_info_prev)
    file.write(describe_info_prev + "\n")

    # Save the first 5000 records to a new CSV
    previous_sample = previous_df.head(5000)
    previous_sample.to_csv('RuizLoans/Data/previous_application_sample.csv', index=False)
    save_message_prev = " Saved first 5000 records of previous_application.csv to previous_application_sample.csv\n"
    print(save_message_prev)
    file.write(save_message_prev + "\n")

print("\n All results saved to RuizLoans/Data/describe.txt")









