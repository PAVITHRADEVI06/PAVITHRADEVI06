from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "http://www.cse.iitm.ac.in/listpeople.php?arg=MSQw"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
names = soup.find('tbody' == 0).find_all('tr', style="padding-top:10px")
for nam in names:
    name = nam.find('td',width ="400").a.text
    Email = nam.find('td',width ="400").span.text

    print(Email)



