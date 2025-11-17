import os
import smtplib
from typing import List, Optional

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import dotenv

dotenv.load_dotenv()

USERNAME = os.getenv("EMAIL_ADDRESS")
PASSWORD = os.getenv("EMAIL_PASSWORD")


def email_text(
    subject: str,
    body_text: str,
    address: str,
    attachment_paths: Optional[List[str]] = None,
) -> dict:
    """
    Send an email with optional PDF attachments.
    If an email is not provided, please prompt the user for one.

    attachment_paths: list of absolute file paths to PDFs
    on the same machine (e.g. /scratch/akshit.kumar/pdfs/sources/....pdf).
    Only give the path not the URL.

    """
    if not USERNAME or not PASSWORD:
        raise RuntimeError("Missing EMAIL_ADDRESS or EMAIL_PASSWORD env vars")

    attachment_paths = attachment_paths or []

    # multipart container (text + attachments)
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["To"] = address
    msg["From"] = USERNAME

    # plain text body
    msg.attach(MIMEText(body_text, "plain"))

    # attach each file
    for path in attachment_paths:
        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
        except FileNotFoundError:
            print(f"[email_text] File not found, skipping attachment: {path}")
            continue
        except Exception as e:
            print(f"[email_text] Error reading {path}: {e}")
            continue

        filename = os.path.basename(path) or "attachment.pdf"

        part = MIMEApplication(pdf_bytes, _subtype="pdf")
        part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(part)

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(USERNAME, PASSWORD)
        smtp.sendmail(msg["From"], [address], msg.as_string())

    return {"status": "sent", "attachments": len(attachment_paths)}


if __name__ == "__main__":
    # quick manual test
    test_pdf = "/scratch/akshit.kumar/pdfs/sources/1540836_Bombay High Court.pdf"

    resp = email_text(
        subject="Test with local PDF",
        body_text="Attaching one local PDF from /scratch.",
        address="akshit.kumar@research.iiit.ac.in",
        attachment_paths=[test_pdf],
    )
    print(resp)
