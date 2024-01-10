from bs4 import BeautifulSoup
import csv
import urllib.request
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context
# Scrape the website to get the data you need

request = urllib.Request(url=url)
kw = dict()
if url.startswith('https://www.iitm.ac.in/faculty'):
    certifi_context = ssl.create_default_context(cafile=certifi.where())
    kw.update(context=certifi_context)
urllib.urlopen(request, **kw)
soup = BeautifulSoup(response.text, 'html.parser')
data = []

for item in soup.find_all('div', {'class': 'item'}):
    name = item.find('span', {'class': 'name'}).text
    email = item.find('a', {'class': 'email'}).text
    data.append({'name': name, 'email': email})
    


