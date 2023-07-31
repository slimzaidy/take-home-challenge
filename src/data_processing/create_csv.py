import os 
import pandas as pd


DATASET_RAW_PATH = \
    os.path.join("data", "raw", "CaliforniaHousing", "cal_housing.data")
OUTPUT_DIR = os.path.join("data", "csv", "CaliforniaHousing")
OUTPUT_FILE = "california_housing.csv"

COLUMN_NAMES = ['Longitude', 'Latitude', 'Housing_median_age', 
                'Total_rooms','Total_bedrooms', 'Population', 
                'Households', 'Median_income','Median_house_value']

def convert_data_to_csv(input_file=DATASET_RAW_PATH,
                        output_dir=OUTPUT_DIR, 
                        output_file=OUTPUT_FILE, 
                        sep=','):
    """
    Convert the raw data file to a CSV file with specified column names.

    Parameters:
        input_file (str): Path to the raw data file.
        output_dir (str): Directory where the CSV file will be saved.
        output_file (str): Filename of the CSV file.
        sep (str): Separator used in the raw data file (default: ',')

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file, sep=sep, header=None)
    df.columns = COLUMN_NAMES
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    convert_data_to_csv()