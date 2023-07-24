import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import schedule
import time
import os
from dotenv import load_dotenv

load_dotenv()


class EmailService:
    def __init__(self):
        self.address = os.getenv("Email")
        self.password = os.getenv("Password")
        self.sendgrid_key = os.getenv("Sendgrid_key")
        
    def send(self, to_address, subject, message):
        
        message1 = Mail(
            from_email=self.address,
            to_emails=to_address,
            subject= subject,
            html_content=message)

        try:
            #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
            sg = SendGridAPIClient(self.sendgrid_key)
            response = sg.send(message1)

            print(response.status_code)
            print(response.body)
            print(response.headers)

        except Exception as e:
            print(e.message)

    def schedule_send(self, to_address:str, subject:str, message:str, hour:str, day:int):
        """
        Ej: 
            es = EmailService()
            es.schedule_send("abc@gmail.com", "Test1", "Message test 1", "08:00", 1) #Send message every Monday at 8:00am
        """

        days = {1:schedule.every().monday, 2:schedule.every().tuesday, 3:schedule.every().wednesday, 4:schedule.every().thursday, 5:schedule.every().friday, 6:schedule.every().saturday, 7:schedule.every().sunday}
        days[day].at(hour).do(self.send, to_address, subject, message)
 
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == '__main__':

    es = EmailService()
    es.send("abelby14@gmail.com", "Test1", "Message test 1")
    es.schedule_send("abelby14@gmail.com", "Test1", "Message test 1", "12:00", 3) #Send message every Mondat at 8:00am