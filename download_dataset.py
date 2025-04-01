import os
import pandas as pd
import requests
from tqdm import tqdm


def download_dataset():
    """Download a phishing dataset from GitHub"""

    # URL for a common phishing dataset
    url = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
    phishing_file = "phishing_urls.txt"

    print(f"Downloading phishing URLs from {url}...")
    response = requests.get(url)

    if response.status_code == 200:
        with open(phishing_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded {len(response.text.splitlines())} phishing URLs")
    else:
        print(f"Failed to download phishing URLs. Status code: {response.status_code}")
        return

    # Get legitimate URLs from Alexa Top sites
    alexa_url = "https://raw.githubusercontent.com/mitchellkrogza/Top-Alexa-Domains/master/top1000-alexa-domains.txt"
    legitimate_file = "legitimate_urls.txt"

    print(f"Downloading legitimate URLs from {alexa_url}...")
    response = requests.get(alexa_url)

    if response.status_code == 200:
        with open(legitimate_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded {len(response.text.splitlines())} legitimate URLs")
    else:
        print(f"Failed to download legitimate URLs. Status code: {response.status_code}")
        return

    # Create combined dataset
    print("Creating combined dataset...")

    # Load phishing URLs (label = 0)
    with open(phishing_file, "r", encoding="utf-8") as f:
        phishing_urls = [line.strip() for line in f if line.strip()]

    # Load legitimate URLs (label = 1)
    with open(legitimate_file, "r", encoding="utf-8") as f:
        legitimate_urls = [f"https://{line.strip()}" for line in f if line.strip()]

    # Limit to manageable size (optional)
    max_urls = min(len(phishing_urls), len(legitimate_urls), 5000)  # Adjust as needed
    phishing_urls = phishing_urls[:max_urls]
    legitimate_urls = legitimate_urls[:max_urls]

    # Create DataFrame
    data = []
    for url in phishing_urls:
        data.append({"url": url, "label": 0})

    for url in legitimate_urls:
        data.append({"url": url, "label": 1})

    df = pd.DataFrame(data)

    # Save to CSV
    output_file = "phishing_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset created with {len(df)} URLs and saved to {output_file}")

    # Clean up
    os.remove(phishing_file)
    os.remove(legitimate_file)

    return output_file


if __name__ == "__main__":
    download_dataset()