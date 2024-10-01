zip_url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
from urllib.request import urlretrieve
if not os.path.exists("zip.test.gz"):
    print("Downloading!")
    urlretrieve(zip_url, "zip.test.gz")

import pandas
