#!/usr/bin/env python
"""
Setup script for Gmail API integration.

This script handles the OAuth flow for Gmail API access by:
1. Creating a .secrets directory if it doesn't exist
2. Using credentials from .secrets/secrets.json to authenticate
3. Opening a browser window for user authentication
4. Storing the access token in .secrets/token.json
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path for imports to work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

# Import required Google libraries
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

def main():
    """Run Gmail authentication setup."""
    # Load environment variables from .env if present
    load_dotenv()
    # Create .secrets directory
    secrets_dir = Path(__file__).parent.absolute() / ".secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for secrets.json or GMAIL_SECRET env var and materialize if present
    secrets_path = secrets_dir / "secrets.json"
    if not secrets_path.exists():
        gmail_secret_env = os.getenv("GMAIL_SECRET")
        if gmail_secret_env:
            try:
                # Accept both raw JSON string or already-parsed JSON via shell export
                secret_obj = json.loads(gmail_secret_env) if isinstance(gmail_secret_env, str) else gmail_secret_env
                with open(secrets_path, "w") as f:
                    json.dump(secret_obj, f)
                print(f"Wrote GMAIL_SECRET from environment to {secrets_path}")
            except Exception as e:
                print(f"Failed to parse GMAIL_SECRET. Ensure it is a JSON object. Error: {str(e)}")
                return 1
        else:
            print(f"Error: Client secrets file not found at {secrets_path}")
            print("Either set GMAIL_SECRET in your environment/.env with the JSON client, or")
            print("download your OAuth client ID JSON from Google Cloud Console and save it as .secrets/secrets.json")
            return 1
    
    print("Starting Gmail API authentication flow...")
    print("A browser window will open for you to authorize access.")
    
    # This will trigger the OAuth flow and create token.json
    try:
        # Define the scopes we need
        SCOPES = [
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/calendar'
        ]
        
        # Load client secrets
        with open(secrets_path, 'r') as f:
            client_config = json.load(f)
        
        # Create the flow using the client_secrets.json format
        flow = InstalledAppFlow.from_client_secrets_file(
            str(secrets_path),
            SCOPES
        )
        
        # Run the OAuth flow
        credentials = flow.run_local_server(port=0)
        
        # Save the credentials to token.json
        token_path = secrets_dir / "token.json"
        token_data = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes,
            'universe_domain': 'googleapis.com',
            'account': '',
            'expiry': credentials.expiry.isoformat() + "Z"
        }
        
        with open(token_path, 'w') as token_file:
            json.dump(token_data, token_file)
            
        print("\nAuthentication successful!")
        print(f"Access token stored at {token_path}")
        print("You can also set GMAIL_TOKEN in your environment to the contents of this token.json if deploying.")
        return 0
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())