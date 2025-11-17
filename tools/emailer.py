import smtplib
from email.mime.text import MIMEText
import dotenv
import os

dotenv.load_dotenv()


USERNAME = os.getenv("EMAIL_ADDRESS")
PASSWORD = os.getenv("EMAIL_PASSWORD")


def email_text(subject: str, body_text: str, address: str):
    """Send an email to the specified address."""
    if not USERNAME or not PASSWORD:
        raise RuntimeError("Missing EMAIL_ADDRESS or EMAIL_PASSWORD env vars")

    msg = MIMEText(body_text)
    msg["Subject"] = subject
    msg["To"] = address
    msg["From"] = USERNAME

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(USERNAME, PASSWORD)
        smtp.sendmail(msg["From"], [address], msg.as_string())

    return {"status": "sent"}


if __name__ == "__main__":
    email("test subject", "this is the body", "akshit.kumar@research.iiit.ac.in")
