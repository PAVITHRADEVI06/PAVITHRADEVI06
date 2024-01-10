import requests
from bs4 import BeautifulSoup

url = 'https://www.iitm.ac.in/faculty'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all the links on the page
links = soup.find_all('a')

# Print the links
for link in links:
    print(link.get('href'))
