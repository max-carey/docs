import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Constants
BASE_URL = "https://linguisticamexicana-amla.colmex.mx"
ARCHIVE_URL = f"{BASE_URL}/index.php/Linguistica_mexicana/issue/archive"
DOWNLOADS_DIR = "downloads"

def ensure_downloads_dir():
    """Create downloads directory if it doesn't exist."""
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR)

def get_issue_links():
    """Get all issue page links from the archive page."""
    print("Fetching archive page...")
    response = requests.get(ARCHIVE_URL)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    issue_links = []
    
    # Find all issue links (they are typically h2 or h3 elements with links)
    for heading in soup.find_all(['h2', 'h3']):
        link = heading.find('a')
        if link and 'href' in link.attrs:
            issue_links.append(link['href'])
    
    return issue_links

def get_pdf_link(issue_url):
    """Get PDF download link from an issue page."""
    print(f"Fetching issue page: {issue_url}")
    response = requests.get(issue_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Try the original PDF link format
    pdf_link = soup.find('a', class_='obj_galley_link pdf')
    if pdf_link and 'href' in pdf_link.attrs:
        return pdf_link['href']
    
    # Try the download link format
    pdf_link = soup.find('a', class_='download')
    if pdf_link and 'href' in pdf_link.attrs:
        return pdf_link['href']
    
    # Try finding the direct download URL from the issue view URL
    issue_id_match = re.search(r'/view/(\d+)(?:/\d+)?$', issue_url)
    if issue_id_match:
        issue_id = issue_id_match.group(1)
        # Construct direct download URL
        download_url = f"{BASE_URL}/index.php/Linguistica_mexicana/issue/download/{issue_id}/56"
        # Verify the URL is valid
        test_response = requests.head(download_url)
        if test_response.status_code == 200:
            return download_url
    
    return None

def download_pdf(pdf_url, issue_number):
    """Download PDF file and save it to the downloads directory."""
    print(f"Downloading PDF from: {pdf_url}")
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': BASE_URL
    }
    
    response = requests.get(pdf_url, stream=True, headers=headers, allow_redirects=True)
    response.raise_for_status()
    
    # Check if we got a PDF
    content_type = response.headers.get('content-type', '').lower()
    if 'application/pdf' not in content_type and 'pdf' not in content_type:
        print(f"Warning: Response may not be a PDF (content-type: {content_type})")
        # If we got HTML, try to extract the direct PDF link
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_script = soup.find('script', text=re.compile('pdfUrl'))
            if pdf_script:
                pdf_url_match = re.search(r'pdfUrl\s*=\s*"([^"]+)"', pdf_script.string)
                if pdf_url_match:
                    return download_pdf(pdf_url_match.group(1), issue_number)
        return False
    
    # Extract issue number from URL or use sequential number
    filename = f"issue_{issue_number}.pdf"
    filepath = os.path.join(DOWNLOADS_DIR, filename)
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Saved PDF to: {filepath}")
    return True

def main():
    """Main function to orchestrate the scraping process."""
    ensure_downloads_dir()
    
    try:
        # Get all issue links
        issue_links = get_issue_links()
        print(f"Found {len(issue_links)} issues")
        
        # Process each issue
        for i, issue_url in enumerate(issue_links, 1):
            try:
                # Get PDF link from issue page
                pdf_url = get_pdf_link(issue_url)
                if pdf_url:
                    # Download the PDF
                    if not download_pdf(pdf_url, i):
                        print(f"Failed to download PDF from: {pdf_url}")
                else:
                    print(f"No PDF link found for issue: {issue_url}")
            except Exception as e:
                print(f"Error processing issue {issue_url}: {str(e)}")
                continue
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
