import requests
from bs4 import BeautifulSoup

# Define the URL of the IITM faculty directory page
url = 'http://www.cse.iitm.ac.in/listpeople.php?arg=MSQw'

# Send a GET request to the URL and get the HTML response
response = requests.get(url)
html_content = response.content

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find all the div elements with class 'staff-details' which contain the professor information
professor_divs = soup.find_all('div', class_='staff-details')

# Loop through the professor divs and extract the research areas and email IDs
for professor_div in professor_divs:
    # Find the h4 element which contains the professor name and extract the text
    professor_name = professor_div.find('h4').text.strip()
    
    # Find the p element with class 'areas' which contains the research areas and extract the text
    research_areas = professor_div.find('p', class_='areas').text.strip()
    
    # Find the a element with class 'email' which contains the email ID and extract the href attribute
    email_link = professor_div.find('a', class_='email')
    if email_link:
        email_id = email_link.get('href').split(':')[1]
    else:
        email_id = None
    
    # Print the professor name, research areas, and email ID
    print('Professor Name:', professor_name)
    print('Research Areas:', research_areas)
    print('Email ID:', email_id)
    print()
