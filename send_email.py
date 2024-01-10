import base64 #converts text data(byte-like objects)into ascii characters
from email.mime.text import MIMEText #used to send text emails
from google_auth_oauthlib.flow import InstalledAppFlow #google_auth performs flow which is basically a series of requests and reponses between the user, resource server and authentication server
from googleapiclient.discovery import build
from requests import HTTPError
import mimetypes #helps in identifying the mime type and encoding of an URL in python

SCOPES = [ "https://www.googleapis.com/auth/gmail.send" ]
flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES) #requests scope using credentials
creds = flow.run_local_server(port=0) #automatically opens url and directs to the local webserver for authentication (port 0 is dynamic port which tells the os to open any port)

service = build('gmail', 'v1', credentials=creds) # defines servicename, serviceversion and localserver 

interest = "Digital Signal Processing, Micro Controllers, Embedded Electronics, Robotics"
body = "Dear Prof \n\nI am Pavithra Devi, a third-year undergraduate student from the Electronics and Communication Engineering department, Panimalar Engineering College. I have been following your course in" + interest + "and would like to work on a research statement/paper as an intern in your group during the summer of 2023 for a period of a month or more\n\n. I have good experience using Verilog HDL, Xilinx, P Spice, and MATLAB. I am also very keen on learning new concepts and using different tools.\n\nProjects: \n\nI'm an Enthusiast who wishes to work on projects that could be impactful to society. \n\n'Voice Command Bot' This bot has multiple functions such as following directions using commands from an app on the mobile phone and it has an ultrasonic sensor that detects obstacles on its own. \n\n'Implementation of Cocktail Party Problem using Singular Value Decomposition' This project takes sounds recorded from two different microphones and separates the different frequencies using principal component analysis. \n\n'Minesweeper Game using Python' Worked on a minesweeper game using the Tkinter library in Python. It is based on object-oriented coding. The Tkinter module is used to develop the graphical user interface (GUI) of the game. \n\nI had briefed about my other projects in the enclosed Resume. \n\nAs I see myself only as a student who will pursue research in the future I will consider it a privilege given an opportunity to contribute to a research project under your group. I hope it will be possible. \n\nI have attached my RESUME for your kind reference. \n\nLooking forward to working with you. Thanking you. \n\nSincere regards,\nPavithra Devi"

message = MIMEText(body) #using mimetext to write the body of the mail

message['to'] = 'brameshbabu@wpi.edu' #mail address of the sender
message['subject'] = 'Regarding internship opportunity under your guidance' #subject 
create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()} 

try:
    message = (service.users().messages().send(userId="me", body=create_message).execute())#service is defined by the scope and the message is sent
    print(F'sent message to {message} Message Id: {message["id"]}')
except HTTPError as error: #http data was unexpected or invalid  
    print(F'An error occurred: {error}')
    message = None




#installedappflow = used for applications are installed in the computer locally.
#googleapiclient is used to acquire the scope using credentials.
#scope = defines the resources that can be accessed by the app. here the app can only 'send' mails it cannot access the mailbox read or modify emails.
#define scope authorization, acquire credentials in google oauth 2.0
#flow:
    #user => app authentication => google
    #user => requests scope using credentials => google
    #user <= approves and returns access token with acknowleged scope <= google (access token = allows the app to access api and perform tasks specified by the scope)
    #user => invokes api, access token and performs tasks or acquires resources => google
#base64.urlsafe_b64encode = encodes the message which is now in bytes(due to the usage of(message.as_bytes))to ascii characters then decode it to unicode string which is the encoding expected by gmail api 
