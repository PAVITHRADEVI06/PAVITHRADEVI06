from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "http://www.cse.iitm.ac.in/listpeople.php?arg=MSQw"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
names = soup.find('tbody' == 0).find_all('tr', style="padding-top:10px")
for nam in names:
    name = nam.find('td',width ="400").a.text
    
    info_str = nam.find('td',width ="400").span.text
    info_block = info_str.split("\n")
    info = info_block[3].strip()
    
    # Extract office information
    office_start = info.find("Office :") + len("Office :")
    office_end = info.find("| Phone")
    office = info[office_start:office_end].strip()
    
    # Extract phone information
    phone_start = info.find("| Phone :") + len("| Phone :")
    phone_end = info.find("Email")
    phone = info[phone_start:phone_end].strip()
    
    # Extract email information
    email_start = info.find("Email :") + len("Email :")
    email_end = info.find("Research Interests")
    email = info[email_start:email_end].strip()
    email = email.replace(" [at] ", "@").replace(" [dot] ", ".")
    
    # Extract research interests
    interest_start = info.find("Research Interests :") + len("Research Interests :")
    interest = info[interest_start:].strip()
    
    print("Name:", name.strip())
    print("Office:", office)
    print("Email:", email)
    print("Research Interests:", interest)

    print("---")