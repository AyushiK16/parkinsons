import io
import urllib.request
import zipfile

# Updated official UCI archive zip endpoint
ZIP_URL = "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip"
OUTPUT_FILE = "parkinsons_updrs.data"


def download_and_extract():
  print("Downloading dataset zip from UCI repository...")
  req = urllib.request.Request(
      ZIP_URL,
      headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
  )

  try:
    with urllib.request.urlopen(req) as response:
      zip_buffer = io.BytesIO(response.read())

    with zipfile.ZipFile(zip_buffer) as z:
      # Find parkinsons_updrs.data inside the zip
      if OUTPUT_FILE in z.namelist():
        z.extract(OUTPUT_FILE, ".")
        print(f"Successfully downloaded and extracted '{OUTPUT_FILE}'!")
      else:
        print(f"Files found in zip: {z.namelist()}")
  except Exception as e:
    print(f"Error downloading dataset: {e}")


if __name__ == "__main__":
  download_and_extract()