import pandas as pd


# Safer loading: handle quotes and skip problematic lines
# Assuming you have uploaded the file to the root of your Google Drive ("My Drive").
# UPDATE THE PATH BELOW IF YOUR FILE IS IN A DIFFERENT LOCATION IN GOOGLE DRIVE.

def readFile():
    df = pd.read_csv(
        "./master_data-rabindra-dhant.txt",
        sep=",",
        quotechar='"',
        engine="python",  # slower but handles messy quoting
        on_bad_lines="skip"  # skips any corrupted rows
    )
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    df.head(3)
    return df
