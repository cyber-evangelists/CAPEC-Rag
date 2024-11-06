
import io
import os
from typing import List, Dict
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from loguru import logger
from src.config.config import Config


class GoogleDriveReader:
    """
    A class to interact with Google Drive, authenticate access, and read different types of files.

    Attributes:
        SCOPES (List[str]): The scope required for accessing Google Drive.
        FOLDER_MIME_TYPE (str): The MIME type for Google Drive folders.
        service (googleapiclient.discovery.Resource): The authenticated Google Drive service instance.
        format_config (Dict[str, Dict]): Configuration mapping MIME types to their respective loaders and export MIME types.
    """


    SCOPES = Config.SCOPES
    FOLDER_MIME_TYPE = Config.FOLDER_MIME_TYPE

    def __init__(self):
        """
        Initializes the GoogleDriveReader by authenticating the Google Drive service and setting up the format configuration.
        """
        self.service = self.authenticate_google_drive()
        self.format_config = {
            "application/vnd.google-apps.document": {
                "export_mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "loader": self.read_docx,
            },
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
                "loader": self.read_docx
            },
            "application/vnd.google-apps.presentation": {
                "export_mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "loader": self.read_pptx,
            },
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": {
                "loader": self.read_pptx
            },
            "application/pdf": {"loader": self.read_pdf},
            "text/plain": {"loader": self.read_text},
            "text/csv": {"loader": self.read_text},
        }


    def authenticate_google_drive(self):
        """
        Authenticates and returns a Google Drive service instance.

        Returns:
            googleapiclient.discovery.Resource: An authenticated Google Drive service instance.
        """
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", self.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return build("drive", "v3", credentials=creds)
        




