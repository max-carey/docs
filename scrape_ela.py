import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict, Optional
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
BASE_URL = "https://ela.enallt.unam.mx"
ARCHIVE_URL = f"{BASE_URL}/index.php/ela/issue/archive"
DOWNLOADS_DIR = "downloads/ela_issues"

# Add headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

def ensure_downloads_dir() -> None:
    """Create downloads directory if it doesn't exist."""
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR)

def get_soup(url: str) -> BeautifulSoup:
    """Get BeautifulSoup object from URL with error handling and rate limiting."""
    try:
        time.sleep(1)  # Rate limiting
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None

def extract_year_issue(text: str) -> Dict[str, str]:
    """Extract year and issue number from text."""
    year_match = re.search(r'Año \d+, Núm\. (\d+)[^0-9]*\(([^)]+)\)', text)
    if year_match:
        issue_num = year_match.group(1)
        date = year_match.group(2)
        year = re.search(r'\d{4}', date).group(0) if re.search(r'\d{4}', date) else None
        return {'issue': issue_num, 'year': year}
    return {'issue': None, 'year': None}

def get_issue_links() -> List[Dict[str, str]]:
    """Get all issue links from the archive page."""
    logging.info("Fetching archive page...")
    soup = get_soup(ARCHIVE_URL)
    if not soup:
        return []

    issues = []
    for heading in soup.find_all(['h4']):  # Issues are in h4 tags
        link = heading.find('a')
        if link and 'href' in link.attrs:
            info = extract_year_issue(heading.text.strip())
            if info['issue'] and info['year']:
                issues.append({
                    'url': link['href'],
                    'issue': info['issue'],
                    'year': info['year'],
                    'title': heading.text.strip()
                })
    
    return issues

def get_pdf_download_link(url: str) -> Optional[str]:
    """Get the actual PDF download link from a page."""
    soup = get_soup(url)
    if not soup:
        return None
    
    # Look for the Spanish "Descargar el archivo PDF" link
    download_link = soup.find('a', string=lambda text: text and 'Descargar el archivo PDF' in text)
    if download_link and 'href' in download_link.attrs:
        return urljoin(BASE_URL, download_link['href'])
    return None

def get_article_links(issue_url: str) -> List[Dict[str, str]]:
    """Get all article links from an issue page."""
    logging.info(f"Fetching issue page: {issue_url}")
    soup = get_soup(issue_url)
    if not soup:
        return []

    articles = []
    # First try to find the issue-level PDF link
    issue_pdf = soup.find('a', class_='file')
    if issue_pdf and 'href' in issue_pdf.attrs:
        # Follow the link to get the actual PDF download URL
        viewer_url = urljoin(BASE_URL, issue_pdf['href'])
        pdf_url = get_pdf_download_link(viewer_url)
        if pdf_url:
            articles.append({
                'title': 'Complete Issue',
                'pdf_url': pdf_url
            })
        return articles

    # If no issue-level PDF, look for individual article PDFs
    for article in soup.find_all('div', class_='obj_article_summary'):
        title_elem = article.find('div', class_='title')
        if title_elem:
            title = title_elem.get_text(strip=True)
            # Try both 'pdf' and 'file' classes
            pdf_link = article.find('a', class_=['pdf', 'file'])
            if pdf_link and 'href' in pdf_link.attrs:
                articles.append({
                    'title': title,
                    'pdf_url': pdf_link['href']
                })
    
    return articles

def download_pdf(pdf_url: str, filepath: str) -> bool:
    """Download PDF file and save it to the specified path."""
    if os.path.exists(filepath):
        logging.info(f"File already exists: {filepath}")
        return True

    logging.info(f"Downloading PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url, headers=HEADERS, stream=True)
        response.raise_for_status()

        # Verify we got a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type and 'pdf' not in content_type:
            logging.warning(f"Response may not be a PDF (content-type: {content_type})")
            return False

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logging.info(f"Saved PDF to: {filepath}")
        return True

    except requests.RequestException as e:
        logging.error(f"Error downloading PDF {pdf_url}: {str(e)}")
        return False

def sanitize_filename(filename: str) -> str:
    """Convert string to valid filename."""
    # Replace invalid characters with underscore
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove or replace other problematic characters
    filename = filename.replace('\n', ' ').replace('\r', ' ')
    # Limit length
    return filename[:150]

def main():
    """Main function to orchestrate the scraping process."""
    ensure_downloads_dir()
    
    try:
        # Get all issue links
        issues = get_issue_links()
        logging.info(f"Found {len(issues)} issues")
        
        # Process each issue
        for issue in issues:
            try:
                # Get article links
                articles = get_article_links(issue['url'])
                logging.info(f"Found {len(articles)} articles in issue {issue['issue']}")
                
                # Download each article
                for i, article in enumerate(articles, 1):
                    # Create filename: issue_number.pdf since we're downloading complete issues
                    filename = f"issue_{issue['issue']}.pdf"
                    filepath = os.path.join(DOWNLOADS_DIR, filename)
                    
                    if not download_pdf(article['pdf_url'], filepath):
                        logging.error(f"Failed to download article: {article['title']}")
                
                # Rate limiting between issues
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error processing issue {issue['title']}: {str(e)}")
                continue
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()