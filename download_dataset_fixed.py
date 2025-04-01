import os
import pandas as pd
import requests
import random


def download_dataset():
    """Download phishing URLs and create a legitimate URL list"""

    # URL for phishing dataset
    url = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
    phishing_file = "phishing_urls.txt"

    print(f"Downloading phishing URLs from {url}...")
    response = requests.get(url)

    if response.status_code == 200:
        with open(phishing_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        phishing_count = len(response.text.splitlines())
        print(f"Downloaded {phishing_count} phishing URLs")
    else:
        print(f"Failed to download phishing URLs. Status code: {response.status_code}")
        return

    # Since Alexa top domains is 404, we'll use a list of known legitimate domains
    print("Creating legitimate URLs list...")

    legitimate_domains = [
        "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
        "twitter.com", "instagram.com", "linkedin.com", "netflix.com", "microsoft.com",
        "apple.com", "yahoo.com", "tiktok.com", "reddit.com", "ebay.com",
        "office.com", "twitch.tv", "adobe.com", "spotify.com", "cnn.com",
        "nytimes.com", "bbc.com", "forbes.com", "espn.com", "walmart.com",
        "target.com", "github.com", "stackoverflow.com", "amazon.co.uk", "paypal.com",
        "dropbox.com", "wordpress.com", "tumblr.com", "quora.com", "zoom.us",
        "pinterest.com", "whatsapp.com", "skype.com", "salesforce.com", "oracle.com",
        "ibm.com", "intel.com", "dell.com", "hp.com", "nvidia.com", "amd.com",
        "cisco.com", "adobe.com", "shopify.com", "outlook.com", "airbnb.com",
        "booking.com", "expedia.com", "uber.com", "lyft.com", "doordash.com",
        "grubhub.com", "postmates.com", "yelp.com", "tripadvisor.com", "zillow.com",
        "realtor.com", "bankofamerica.com", "chase.com", "wellsfargo.com", "citibank.com",
        "capitalone.com", "discover.com", "amex.com", "visa.com", "mastercard.com",
        "fedex.com", "ups.com", "usps.com", "dhl.com", "nasa.gov", "whitehouse.gov",
        "cdc.gov", "who.int", "nsa.gov", "fbi.gov", "cia.gov", "irs.gov", "mit.edu",
        "harvard.edu", "stanford.edu", "berkeley.edu", "yale.edu", "columbia.edu",
        "cornell.edu", "princeton.edu", "nyu.edu", "ucla.edu", "uchicago.edu",
        "discord.com", "slack.com", "teams.microsoft.com", "telegram.org", "signal.org",
        "notion.so", "trello.com", "asana.com", "monday.com", "atlassian.com",
    ]

    # Generate more domains by adding subdomains and paths to increase dataset size
    extended_legitimate_urls = []

    # Add base domains with https
    for domain in legitimate_domains:
        extended_legitimate_urls.append(f"https://{domain}")

    # Add common subdomains
    subdomains = ["www", "mail", "blog", "shop", "support", "help", "developer", "docs"]
    for domain in legitimate_domains[:50]:  # Use first 50 domains
        for subdomain in subdomains:
            extended_legitimate_urls.append(f"https://{subdomain}.{domain}")

    # Add paths to some domains
    paths = ["/", "/index.html", "/about", "/contact", "/login", "/products", "/services", "/blog", "/help"]
    for domain in legitimate_domains[:30]:  # Use first 30 domains
        for path in paths:
            extended_legitimate_urls.append(f"https://www.{domain}{path}")

    # Create a more diverse set
    random.shuffle(extended_legitimate_urls)

    # Limit the number to match phishing URLs
    max_urls = min(phishing_count, len(extended_legitimate_urls), 5000)  # Maximum 5000 URLs
    print(f"Using {max_urls} URLs for each category")

    # Create combined dataset
    print("Creating combined dataset...")

    # Load phishing URLs (label = 0)
    with open(phishing_file, "r", encoding="utf-8") as f:
        phishing_urls = [line.strip() for line in f if line.strip()][:max_urls]

    legitimate_urls = extended_legitimate_urls[:max_urls]

    # Create DataFrame
    data = []
    print("Processing phishing URLs...")
    for i, url in enumerate(phishing_urls):
        if i % 500 == 0:
            print(f"Processed {i}/{len(phishing_urls)} phishing URLs")
        data.append({"url": url, "label": 0})

    print("Processing legitimate URLs...")
    for i, url in enumerate(legitimate_urls):
        if i % 500 == 0:
            print(f"Processed {i}/{len(legitimate_urls)} legitimate URLs")
        data.append({"url": url, "label": 1})

    df = pd.DataFrame(data)

    # Save to CSV
    output_file = "phishing_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset created with {len(df)} URLs and saved to {output_file}")

    # Clean up
    os.remove(phishing_file)

    return output_file


if __name__ == "__main__":
    download_dataset()