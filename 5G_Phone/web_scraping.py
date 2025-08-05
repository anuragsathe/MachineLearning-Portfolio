# Import required libraries
import pandas as pd                      # For working with CSV and dataframes
import requests                          # For making HTTP requests to websites
from bs4 import BeautifulSoup            # For parsing and scraping HTML content
import time                              # To add delay between requests
import re                                # For regular expressions (not used here but imported)
import warnings                          # To ignore unnecessary warnings
warnings.filterwarnings('ignore')       # Ignore all warnings (good for clean output)

print('Setup Complete!')

no_pages = 40                            # Total number of pages you want to scrape

# Function to scrape data from one Flipkart page
def get_data(pageNo):
    
    # Setting headers to mimic a browser request (helps to avoid being blocked)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "close",
    }
    
    # Flipkart URL with page number and filters for phones above ₹10,000
    url = f'https://www.flipkart.com/search?q=best+phone&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&p%5B%5D=facets.price_range.from%3D10000&p%5B%5D=facets.price_range.to%3DMax&page={pageNo}'

    # Send request to Flipkart
    response = requests.get(url, headers=headers)
    
    # Check if response is successful
    if response.status_code != 200:
        print(f"Failed to retrieve page {pageNo}. Status Code: {response.status_code}")
        return []
    
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find all phone blocks using the specific class name
    phone_blocks = soup.find_all('div', attrs={'class': 'tUxRFH'})
    
    print(f"Found {len(phone_blocks)} phone items on page {pageNo}")

    alls = []  # To store data for all phones on the page
    for d in phone_blocks:
        all1 = []
        
        # Phone name
        name = d.find('div', attrs={'class':'KzDlHZ'})
        all1.append(name.text.strip() if name else "Not Available")
                   
        # Extract phone specifications
        specs = d.find_all('li', attrs={'class': 'J+igdf'})
        memory    = specs[0].text if len(specs) > 0 else 'Not Available'
        display   = specs[1].text if len(specs) > 1 else 'Not Available'
        camera    = specs[2].text if len(specs) > 2 else 'Not Available'
        battery   = specs[3].text if len(specs) > 3 else 'Not Available'
        processor = specs[4].text if len(specs) > 4 else 'Not Available'
        warranty  = specs[5].text if len(specs) > 5 else "Not Available"
        
        # Add specs to list
        all1 += [memory, display, camera, battery, processor, warranty]

        # Phone rating
        rating = d.find('div', attrs={'class': "XQDdHH"})
        all1.append(rating.text.strip() if rating else "Not Available")
        
        # Discount info
        dis = d.find('div', attrs={'class': "UkUFwK"})
        all1.append(dis.text.strip() if dis else "Not Available")
        
        # Phone price
        price = d.find('div', attrs={'class': "Nx9bqj _4b5DiR"})
        all1.append(price.text.strip() if price else "Not Available")
        
        # Add phone data to the main list
        alls.append(all1)
        
    return alls

# -------------------- Main Execution --------------------

results = []

# Loop through all pages and get phone data
for i in range(1, no_pages + 1):
    page_data = get_data(i)
    results.extend(page_data)
    time.sleep(3)  # 3 second delay – important to avoid getting blocked (server friendly)

# Create pandas DataFrame with all collected data
df = pd.DataFrame(results, columns=['Phone Name','Memory','Display','Camera','Battery','Processor','Warranty','Rating', 'Discount',"Price"])

# Save DataFrame to CSV file
df.to_csv('phone.csv', index=False, encoding='utf-8')
print("Data saved to 'Flipkart_products.csv'")

# Load the CSV to check content
df = pd.read_csv("phone.csv")
print(df.head(18))  # Print first 18 rows of the data
