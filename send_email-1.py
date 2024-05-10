import base64
from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from requests import HTTPError
import mimetypes

SCOPES = [
        "https://www.googleapis.com/auth/gmail.send"
    ]
flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server(port=0)

service = build('gmail', 'v1', credentials=creds)

interest = "Digital Signal Processing, Micro Controllers, Embedded Electronics, Robotics"
body = "Dear Prof \n\nI am Pavithra Devi, a third-year undergraduate student from the Electronics and Communication Engineering department, Panimalar Engineering College. I have been following your research in the domain of  " + interest + " and would like to work on a research statement/paper as an intern in your group during the summer of 2023 for a period of a month or more. \n\nI have good experience using Verilog HDL, Xilinx, P Spice, and MATLAB. I am also very keen on learning new concepts and using different tools.\n\nProjects: \n\nI'm an Enthusiast who wishes to work on projects that could be impactful to society. \n\n'Voice Command Bot' This bot has multiple functions such as following directions using commands from an app on the mobile phone and it has an ultrasonic sensor that detects obstacles on its own. \n\n'Implementation of Cocktail Party Problem using Singular Value Decomposition' This project takes sounds recorded from two different microphones and separates the different frequencies using principal component analysis. \n\n'Minesweeper Game using Python' Worked on a minesweeper game using the Tkinter library in Python. It is based on object-oriented coding. The Tkinter module is used to develop the graphical user interface (GUI) of the game. \n\nI had briefed about my other projects in the enclosed Resume. \n\nAs I see myself only as a student who will pursue research in the future I will consider it a privilege given an opportunity to contribute to a research project under your group. I hope it will be possible. \n\nI have attached my RESUME for your kind reference. \n\nLooking forward to working with you. Thanking you. \n\nSincere regards,\nPavithra Devi"

message = MIMEText(body)

message['to'] = 'kumar7bharath@gmail.com'
message['subject'] = 'Regarding internship opportunity under your guidance'
create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

try:
    message = (service.users().messages().send(userId="me", body=create_message).execute())
    print(F'sent message to {message} Message Id: {message["id"]}')
except HTTPError as error:
    print(F'An error occurred: {error}')
    message = None
