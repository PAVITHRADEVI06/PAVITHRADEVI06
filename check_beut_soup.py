from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "http://www.cse.iitm.ac.in/listpeople.php?arg=MSQw"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
print(soup.get_text())
