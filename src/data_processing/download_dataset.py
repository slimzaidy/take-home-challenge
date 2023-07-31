import os
import tarfile
import urllib.request 

DOWNLOAD_URL = "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz"
LOCAL_DS_PATH = os.path.join("data", "raw")

def get_housing_data(housing_url=DOWNLOAD_URL, housing_path=LOCAL_DS_PATH):
    """
    Download and extract the housing dataset from the given URL.

    Parameters:
        housing_url (str): URL to download the housing data archive (default: DOWNLOAD_URL).
        housing_path (str): Local directory path to store the downloaded data (default: LOCAL_DS_PATH).

    Returns:
        None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(DOWNLOAD_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)


if __name__ == "__main__":
    get_housing_data()