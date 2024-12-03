from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
from typing import List, Dict
import base64
from email.mime.text import MIMEText
import html2text
from typing import List, Union

CREDENTIALS_PATH = 'credentials.json'

class EmailContent(BaseModel):
    subject: str
    sender: str
    recipient: str
    date: str
    body: str

class EmailMetadata(BaseModel):
    id: str
    subject: str
    sender: str
    snippet: str
    date: str
    is_unread: bool
    thread_id: str

class GmailDeps:
    def __init__(self):
        self.service = self.get_gmail_service()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        # Get user profile when initializing
        self.user_profile = self.get_user_profile()
    
    def get_gmail_service(self):
        SCOPES = ['https://mail.google.com/']
        creds = None
        
        if os.path.exists('token.pickle'):
            print("Hittar sparade tokens, använder dem...")
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Uppdaterar utgångna tokens...")
                creds.refresh(Request())
            else:
                print("Behöver ny autentisering, öppnar webbläsare...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
                print("Tokens sparade för framtida användning!")
        
        return build('gmail', 'v1', credentials=creds)
    
    def get_user_profile(self) -> dict:
        """Get the user's Gmail profile information."""
        try:
            profile = self.service.users().getProfile(userId='me').execute()
            print(f"Inloggad som: {profile['emailAddress']}")
            return profile
        except Exception as e:
            print(f"Kunde inte hämta användarprofil: {e}")
            return {"emailAddress": "unknown@gmail.com"}



gmail_agent = Agent(
    'openai:gpt-4o',
    deps_type=GmailDeps,
    result_type=str,
    system_prompt="""Du är en hjälpsam email-assistent. Du kan:
    1. Läsa och sammanfatta email (både lästa och olästa)
    2. Skicka nya email
    3. Svara på email i samma konversationstråd
    4. Markera email som lästa eller olästa
    5. Söka efter all korrespondens med en specifik emailadress
    
    När du listar email, visa alltid om de är lästa eller olästa.
    
    När du skickar email eller svarar på email, fråga ALLTID om bekräftelse först genom att visa:
    - Mottagare
    - Ämne
    - Meddelande
    
    För korrespondenssökning, presentera resultaten på ett överskådligt sätt med datum.
    
    Var hjälpsam och tydlig."""
)
@gmail_agent.system_prompt
def add_user_context(ctx: RunContext[GmailDeps]) -> str:
    """Add user context to the system prompt."""
    email = ctx.deps.user_profile['emailAddress']
    return f"""Du hanterar Gmail-kontot för {email}.

När du skickar email, kom ihåg att du skickar från {email}.

Om någon ber dig skicka ett mail från en annan adress, förklara att du bara kan skicka från {email}."""

@gmail_agent.tool
def list_recent_emails(ctx: RunContext[GmailDeps], max_results: int = 3, unread_only: bool = False) -> List[dict]:
    """Hämta de senaste emailen.
    
    Args:
        max_results: Antal email att hämta
        unread_only: Om True, visa endast olästa mail
    """
    query = 'is:unread' if unread_only else ''
    
    results = ctx.deps.service.users().messages().list(
        userId='me', maxResults=max_results, q=query
    ).execute()
    
    messages = results.get('messages', [])
    email_summaries = []
    
    for message in messages:
        msg = ctx.deps.service.users().messages().get(
            userId='me', id=message['id']
        ).execute()
        
        headers = msg['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'Inget ämne')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Okänd avsändare')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Inget datum')
        
        # Check if message is unread
        is_unread = 'UNREAD' in msg['labelIds'] if 'labelIds' in msg else False
        
        email_summaries.append({
            "id": message['id'],
            "thread_id": msg['threadId'],
            "subject": subject,
            "sender": sender,
            "date": date,
            "snippet": msg['snippet'],
            "is_unread": is_unread
        })
    
    return email_summaries

@gmail_agent.tool
def search_correspondence(ctx: RunContext[GmailDeps], email_address: str, max_results: int = 10) -> List[dict]:
    """Sök efter all korrespondens med en specifik emailadress.
    
    Args:
        email_address: Emailadressen att söka efter
        max_results: Max antal resultat att returnera
    """
    print(f"Söker efter korrespondens med {email_address}...")
    
    query = f"from:{email_address} OR to:{email_address}"
    results = ctx.deps.service.users().messages().list(
        userId='me', maxResults=max_results, q=query
    ).execute()
    
    messages = results.get('messages', [])
    correspondence = []
    
    for message in messages:
        msg = ctx.deps.service.users().messages().get(
            userId='me', id=message['id']
        ).execute()
        
        headers = msg['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'Inget ämne')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Okänd avsändare')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Inget datum')
        
        correspondence.append({
            "id": message['id'],
            "thread_id": msg['threadId'],
            "subject": subject,
            "sender": sender,
            "date": date,
            "snippet": msg['snippet']
        })
    
    return correspondence

@gmail_agent.tool
def mark_as_read(ctx: RunContext[GmailDeps], email_id: str) -> str:
    """Markera ett email som läst.
    
    Args:
        email_id: ID för emailet som ska markeras som läst
    """
    try:
        ctx.deps.service.users().messages().modify(
            userId='me',
            id=email_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        return "Email markerat som läst"
    except Exception as e:
        return f"Fel vid markering av email: {str(e)}"

@gmail_agent.tool
def mark_as_unread(ctx: RunContext[GmailDeps], email_id: str) -> str:
    """Markera ett email som oläst.
    
    Args:
        email_id: ID för emailet som ska markeras som oläst
    """
    try:
        ctx.deps.service.users().messages().modify(
            userId='me',
            id=email_id,
            body={'addLabelIds': ['UNREAD']}
        ).execute()
        return "Email markerat som oläst"
    except Exception as e:
        return f"Fel vid markering av email: {str(e)}"
    
@gmail_agent.tool
def read_email(ctx: RunContext[GmailDeps], email_id: str) -> Dict:
    """Läs hela innehållet i ett specifikt email.
    
    Args:
        email_id: ID för emailet som ska läsas
    """
    print(f"Läser email med ID: {email_id}")
    
    # First read the email
    message = ctx.deps.service.users().messages().get(
        userId='me', 
        id=email_id, 
        format='full'
    ).execute()
    
    # Mark as read immediately after reading
    try:
        ctx.deps.service.users().messages().modify(
            userId='me',
            id=email_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        print("Email markerat som läst")
    except Exception as e:
        print(f"Kunde inte markera email som läst: {str(e)}")

    headers = message['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'Inget ämne')
    sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Okänd avsändare')
    recipient = next((h['value'] for h in headers if h['name'] == 'To'), 'Ingen mottagare')
    date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Inget datum')
    
    # Extrahera emailets innehåll
    if 'parts' in message['payload']:
        parts = message['payload']['parts']
        body = ""
        for part in parts:
            if part.get('mimeType') == 'text/plain':
                if 'data' in part['body']:
                    body = base64.urlsafe_b64decode(part['body']['data']).decode()
                    break
            elif part.get('mimeType') == 'text/html':
                if 'data' in part['body']:
                    html_content = base64.urlsafe_b64decode(part['body']['data']).decode()
                    body = ctx.deps.html_converter.handle(html_content)
                    break
    else:
        body_data = message['payload']['body'].get('data', '')
        if body_data:
            body = base64.urlsafe_b64decode(body_data).decode()
        else:
            body = 'Inget innehåll'
    
    return {
        "subject": subject,
        "sender": sender,
        "recipient": recipient,
        "date": date,
        "body": body
    }
@gmail_agent.tool
def send_email(
    ctx: RunContext[GmailDeps], 
    to: Union[str, List[str]], 
    subject: str, 
    body: str,
    cc: Union[str, List[str]] = None
) -> str:
    """Skicka ett email.
    
    Args:
        to: En mottagare eller lista av mottagare
        subject: Ämnesrad
        body: Email-meddelandet
        cc: CC-mottagare eller lista av CC-mottagare (valfritt)
    """
    print(f"Förbereder att skicka email...")
    
    # Konvertera enskilda mottagare till listor
    to_list = [to] if isinstance(to, str) else to
    cc_list = [cc] if isinstance(cc, str) and cc else [] if cc is None else cc
    
    message = MIMEText(body)
    message['to'] = ', '.join(to_list)
    message['subject'] = subject
    
    if cc_list:
        message['cc'] = ', '.join(cc_list)
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    try:
        sent_message = ctx.deps.service.users().messages().send(
            userId='me', 
            body={'raw': raw}
        ).execute()
        return f"Email skickat framgångsrikt! Message ID: {sent_message['id']}"
    except Exception as e:
        return f"Fel vid skickande av email: {str(e)}"

@gmail_agent.tool
def reply_to_email(
    ctx: RunContext[GmailDeps], 
    email_id: str, 
    body: str,
    cc: Union[str, List[str]] = None,
    additional_to: Union[str, List[str]] = None
) -> str:
    """Reply to a specific email, maintaining the conversation thread.
    
    Args:
        email_id: ID of the email to reply to
        body: The reply message content
        cc: Additional CC recipients (optional)
        additional_to: Additional direct recipients (optional)
    """
    print(f"Förbereder svar på email med ID: {email_id}")
    
    # Get the original message
    original = ctx.deps.service.users().messages().get(
        userId='me', id=email_id, format='metadata',
        metadataHeaders=['Subject', 'From', 'To', 'Cc', 'Message-ID', 'References', 'In-Reply-To']
    ).execute()
    
    headers = original['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
    if not subject.lower().startswith('re:'):
        subject = f"Re: {subject}"
    
    # Get original recipients
    from_header = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
    original_to = next((h['value'] for h in headers if h['name'].lower() == 'to'), '')
    original_cc = next((h['value'] for h in headers if h['name'].lower() == 'cc'), '')
    
    # Extract primary recipient (original sender)
    to_address = from_header.split('<')[-1].rstrip('>')
    
    # Create lists of all recipients
    to_list = [to_address]
    if additional_to:
        additional_to_list = [additional_to] if isinstance(additional_to, str) else additional_to
        to_list.extend(additional_to_list)
    
    # Handle CC recipients
    cc_list = []
    if cc:
        new_cc_list = [cc] if isinstance(cc, str) else cc
        cc_list.extend(new_cc_list)
    
    # Create the message
    message = MIMEText(body)
    message['to'] = ', '.join(to_list)
    message['subject'] = subject
    
    if cc_list:
        message['cc'] = ', '.join(cc_list)
    
    # Add threading headers
    original_message_id = next(
        (h['value'] for h in headers if h['name'].lower() == 'message-id'),
        None
    )
    
    if original_message_id:
        message['In-Reply-To'] = original_message_id
        references = next(
            (h['value'] for h in headers if h['name'].lower() == 'references'),
            ''
        )
        if references:
            message['References'] = f"{references} {original_message_id}"
        else:
            message['References'] = original_message_id
    
    # Add thread ID if available
    thread_id = original.get('threadId')
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    try:
        sent_message = ctx.deps.service.users().messages().send(
            userId='me',
            body={'raw': raw, 'threadId': thread_id}
        ).execute()
        recipients_summary = (
            f"To: {', '.join(to_list)}\n"
            f"CC: {', '.join(cc_list) if cc_list else 'Ingen'}"
        )
        return f"Svar skickat framgångsrikt!\n{recipients_summary}\nMessage ID: {sent_message['id']}"
    except Exception as e:
        return f"Fel vid skickande av svar: {str(e)}"
    
    
    
if __name__ == "__main__":
    print("Startar Gmail-assistenten...")
    try:
        deps = GmailDeps()
        message_history = []
        
        while True:
            user_input = input("\nVad vill du göra? (Skriv 'avsluta' för att avsluta)\n> ")
            
            if user_input.lower() == 'avsluta':
                break
                
            result = gmail_agent.run_sync(
                user_input, 
                deps=deps,
                message_history=message_history
            )
            
            message_history.extend(result.new_messages())
            
            print("\nAI Assistentens svar:")
            print(result.data)
            
    except Exception as e:
        print(f"Ett fel uppstod: {str(e)}")
        print("Feldetaljer:", str(e))
    
    print("\nAvslutar Gmail-assistenten...")