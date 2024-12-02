from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
from lxml import etree
import re
import os
import requests
from urllib.parse import urlparse
from pathlib import Path

SCROLL_PAUSE_TIME = 2
SCROLL_TIMES = 20

def configure_driver():
    """Configure and return a Selenium Chrome driver with appropriate options."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Headless mode (optional)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(options=options)

def extract_image_paths(dom):
    """Extract image paths from the DOM using XPath."""
    image_paths = []
    positive_class_ = "hCL"
    negative_class_ = "XiG"
    element_with_class = dom.xpath(f"//*[contains(@class, '{positive_class_}') and not(contains(@class, '{negative_class_}'))]")
    for element in element_with_class:
        image_paths.append(element.attrib["src"])
    return image_paths

def scroll_and_extract(driver):
    """Scroll the page and extract image paths."""
    image_paths = []
    
    # Get initial content
    html_content = driver.page_source
    dom = etree.HTML(html_content)
    image_paths.extend(extract_image_paths(dom))
    
    # Scroll and get more content
    for _ in range(SCROLL_TIMES):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 1.3);")
        time.sleep(SCROLL_PAUSE_TIME)
        
        html_content = driver.page_source
        dom = etree.HTML(html_content)
        image_paths.extend(extract_image_paths(dom))
    
    return image_paths

def process_image_paths(paths):
    """Process and clean image paths."""
    unique_paths = list(set(paths))
    return [re.sub(r'/\d+x/', '/736x/', path) for path in unique_paths]

def download_images(dir_name, paths):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    if paths == []:
        print("No new images to download!")
        return
        
    for i, img_url in enumerate(paths):
        try:
            response = requests.get(img_url)
            
            # Skip if the request was unsuccessful
            if response.status_code != 200:
                print(f"Failed to download image {i+1}: HTTP {response.status_code}")
                continue
                
            # Extract file extension from URL or default to .jpg
            file_ext = Path(urlparse(img_url).path).suffix
            if not file_ext:
                file_ext = '.jpg'
                
            # Create filename
            filename = f'{dir_name}/{img_url.split("/")[-1].split(".")[0]}{file_ext}'
            
            # Save the image
            with open(filename, 'wb') as f:
                f.write(response.content)
                
            print(f"Downloaded image {i+1}/{len(paths)}", end='\r')
            
        except Exception as e:
            print(f"Error downloading image {i+1}: {str(e)}")

    print("\nDownload complete!\n")


if __name__ == "__main__":
    # Configure and initialize the driver
    driver = configure_driver()
    
    # searches = [
    #     "porsche", "lamborghini", "mclaren", "ferrari", "aston martin", "ford", "mercedes benz", "bmw",
    #     "audi", "maserati", "jaguar car", "bentley", "volkswagen", "volvo", "mini cooper", "fiat",
    #     "renault", "peugeot", "citroen", "opel", "hyundai", "kia", "land rover", "lexus", "nissan",
    #     "lotus car"
    # ]
    
    # Faltan: searches = ["tesla", "chevrolet"]
    
    for search in searches:
        print(f"Searching for {search}")
        # Load the Pinterest page
        url = f"https://es.pinterest.com/search/pins/?q={search}"
        driver.get(url)
        
        # Wait for JavaScript to render the page
        time.sleep(5)
        
        image_paths = scroll_and_extract(driver)
        image_paths = process_image_paths(image_paths)
        
        downloaded_images = [] # List of downloaded images to avoid duplicates
        for root, dirs, files in os.walk('downloaded_images'):
            downloaded_images.extend(files)
        downloaded_images = [img.split('.')[0] for img in downloaded_images]
        image_paths = [path for path in image_paths if path.split('/')[-1].split('.')[0] not in downloaded_images]
        download_images(dir_name=f'downloaded_images/{search}', paths=image_paths)
    
    # Clean up
    driver.quit()
