import time 
from fastapi import FastAPI, HTTPException, Body

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import logging
import google.generativeai as genai
import os
from typing import List, Optional, Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from datetime import datetime
import gspread
from google.oauth2 import service_account
from datetime import datetime
import json
import tempfile

app = FastAPI(title="Facebook Comment Reply Bot with Scheduler")

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Store the current scheduled job
current_job = None
is_running = False

# Preset replies, initially with some default keys and responses
preset_replies = {
    "hi there": "Hi there too!",
    "not so good": "We're sorry to hear that. Could you let us know what we can improve?",
    "how are you": "I'm doing well! How about you?",
    "rambunctious dinosaur": "That sounds like a wild dinosaur!",
    "শুভ কামনা রইলো": "Thank you!"
}

class FacebookConfig(BaseModel):
    page_id: str
    access_token: str

class GoogleCredentials(BaseModel):
    credentials: dict  # User-provided Google API credentials as a JSON object
    sheet_name: str = "Sheet1"  # Default to Sheet1 if not specified

class AIReplyRequest(BaseModel):
    config: FacebookConfig
    gemini_api_key: str
    interval_seconds: int  # User-defined interval for scheduling
    google_sheet_id: str  # ID of the Google Sheet
    google_credentials: GoogleCredentials  # User-provided Google credentials

# Function to get Google Sheets client using user-provided credentials
def get_sheets_client(credentials_dict, temp_file=None):
    try:
        # Create a temporary file to store the credentials
        if temp_file is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
            json.dump(credentials_dict, temp_file)
            temp_file.flush()
        
        creds = service_account.Credentials.from_service_account_file(
            temp_file.name, 
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        client = gspread.authorize(creds)
        return client, temp_file
    except Exception as e:
        logger.error(f"Error initializing Google Sheets client: {e}")
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except:
                pass
        raise

def load_existing_replies(sheet_id: str, credentials_dict: dict, sheet_name: str) -> set:
    """Load existing comment IDs from Google Sheets to prevent duplicate replies."""
    temp_file = None
    try:
        client, temp_file = get_sheets_client(credentials_dict)
        try:
            sheet = client.open_by_key(sheet_id).worksheet(sheet_name)
            # Get all comment IDs from the spreadsheet
            values = sheet.get_all_values()
            if len(values) > 1:  # If there's more than just the header row
                # Find the Comment ID column index
                comment_id_idx = values[0].index("Comment ID")
                # Get all comment IDs (skip the header row)
                return set(row[comment_id_idx] for row in values[1:])
            return set()
        except gspread.exceptions.WorksheetNotFound:
            # Create the worksheet if it doesn't exist
            spreadsheet = client.open_by_key(sheet_id)
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1, cols=11)
            # Create header row
            header = ["Post ID", "Post Content", "Post URL", "Post Time", "Comment ID", 
                      "Comment Content", "Comment URL", "Comment Time", "Commenter Name", "Reply"]
            worksheet.update([header])
            return set()
    except Exception as e:
        logger.error(f"Error loading existing replies from Google Sheets: {e}")
        return set()
    finally:
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except:
                pass

def store_data_in_sheet(data: Dict[str, Any], sheet_id: str, credentials_dict: dict, sheet_name: str):
    """Stores comment data along with post content into Google Sheets."""
    temp_file = None
    try:
        client, temp_file = get_sheets_client(credentials_dict)
        try:
            sheet = client.open_by_key(sheet_id).worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            # Create the worksheet if it doesn't exist
            spreadsheet = client.open_by_key(sheet_id)
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1, cols=11)
            # Create header row
            header = ["Post ID", "Post Content", "Post URL", "Post Time", "Comment ID", 
                     "Comment Content", "Comment URL", "Comment Time", "Commenter Name", "Reply"]
            sheet.update([header])

        # Create a row with the data
        row = [
            data.get("Post ID", ""),
            data.get("Post Content", ""),
            data.get("Post URL", ""),
            data.get("Post Time", ""),
            data.get("Comment ID", ""),
            data.get("Comment Content", ""),
            data.get("Comment URL", ""),
            data.get("Comment Time", ""),
            data.get("Commenter Name", ""),
            data.get("Reply", "")
        ]

        # Append the row to the sheet
        sheet.append_row(row)
        
        logger.info("Data stored in Google Sheets successfully.")
    except Exception as e:
        logger.error(f"Error storing data in Google Sheets: {e}")
        raise
    finally:
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except:
                pass

def preset_replies_check(comment: str) -> Optional[str]:
    """Check for preset replies based on keyword matching, allowing 75% word match and ignoring punctuation."""
    global preset_replies  # Ensure we use the updated preset_replies

    # Remove punctuation from the comment and convert it to lowercase
    comment = re.sub(r"[^\w\s]", "", comment).lower()

    # Tokenize the comment into words
    comment_words = set(comment.split())

    for keyword, reply in preset_replies.items():
        # Clean the keyword in the same way as the comment
        clean_keyword = re.sub(r"[^\w\s]", "", keyword).lower()
        keyword_words = set(clean_keyword.split())

        # Check if at least 75% of the words in the keyword match the comment
        if len(keyword_words.intersection(comment_words)) / len(keyword_words) >= 0.75:
            return reply

    return None

def generate_ai_reply(comment: str, gemini_api_key: str, post_message: Optional[str] = None, 
                     preset_reply: Optional[str] = None, commenter_name: Optional[str] = None, 
                     commenter_profile_link: Optional[str] = None) -> str:
    """Generate AI reply using Gemini model while also considering preset replies, commenter name, and profile link."""
    try:
        time.sleep(1)
        genai.configure(api_key=gemini_api_key)
        #model = genai.GenerativeModel('learnlm-1.5-pro-experimental')
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Determine if the comment is in Bengali (simple check based on characters)
        is_bengali = any('\u0980' <= char <= '\u09FF' for char in comment)

        if is_bengali:
            comment_language="bengali"
        else:
            comment_language='others'

        # Create the prompt with the commenter name, profile link, and Bengali instruction if applicable
          # Commenter Name: {commenter_name or 'Anonymous'}
        prompt = f"""
        You are a page moderator.
        Post content: {post_message }
        Comment: {comment}
        Comment language:{comment_language}
      
        Commenter Profile Link: {commenter_profile_link }
        Preset Reply (if any): {preset_reply}

        if preset reply is none then ignore it mostly.
        Generate a polite reply that:
        
        2. Enhance the preset reply if available. If no preset reply exists, generate a relevant response based on the comment and post content."
        3. Is professional and friendly.
        4. Is concise (max 200 characters).
        5. Does not express uncertainty (e.g., "I don't know how to reply").
        6. If the comment is in Bengali, reply in Bengali. If not, reply in English.
        so try to use bengali if comment is in bengali to reply
        don't mix bengali and english in the reply use one language preferably bengali if 
        comment is in bengali
        Do not state explicitly that the comment is not in Bengali. Directly generate a relevant English reply if the comment is not in Bengali
       so generate a reply directly based on post content, comment etc
         """

        # Generate content using the model
        response = model.generate_content(prompt)
        ai_reply = response.text.strip()

        # Return the AI-generated reply with the preset reply included if available
        #return f"{preset_reply} {ai_reply}" if preset_reply else ai_reply
        return ai_reply

    except Exception as e:
        logger.error(f"AI reply generation failed: {e}")
        return f"Thank you for your comment"
    


@app.post("/add-preset-reply")
def add_preset_reply(preset_data: dict):
    """Endpoint to add multiple preset replies using key-value pairs"""
    if not preset_data:
        raise HTTPException(status_code=400, detail="No preset data provided")

    global preset_replies
    new_entries = []
    duplicate_keys = []

    # Validate and process each key-value pair
    for key, reply in preset_data.items():
        if not key or not reply:
            raise HTTPException(status_code=400, detail="Empty key or reply detected")
            
        if key in preset_replies:
            duplicate_keys.append(key)
            continue
            
        preset_replies[key] = reply
        new_entries.append(key)

    if duplicate_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate keys found: {', '.join(duplicate_keys)}. New entries added: {len(new_entries)}"
        )

    logger.info(f"Added {len(new_entries)} new preset replies")
    return {
        "status": "Partial success" if duplicate_keys else "Complete success",
        "added": new_entries,
        "duplicates": duplicate_keys
    }
    












@app.get("/get-sheet-link/{sheet_id}")
async def get_sheet_link(sheet_id: str):
    """Return a direct link to the Google Sheet."""
    return {"sheet_link": f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"}

def get_facebook_posts(page_id: str, access_token: str) -> List[Dict[str, Any]]:
    """Fetch all posts from a Facebook Page, handling pagination."""
    url = f"https://graph.facebook.com/v22.0/{page_id}/feed"
    params = {"access_token": access_token, "fields": "id,message,created_time,permalink_url"}

    posts = []
    
    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Add fetched posts to the list
            posts.extend(data.get('data', []))

            # Check if there is a next page for pagination
            url = data.get('paging', {}).get('next', None)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching posts: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch posts: {str(e)}")

    # Sort posts by created_time (newest first)
    posts.sort(key=lambda x: x.get('created_time', ''), reverse=True)

    return posts

def get_facebook_comments(post_id: str, access_token: str) -> List[Dict[str, Any]]:
    """Fetch comments for a specific post, handling pagination."""
    url = f"https://graph.facebook.com/v22.0/{post_id}/comments"
    params = {"access_token": access_token, "fields": "id,message,created_time,permalink_url,from"}

    comments = []
    
    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Add fetched comments to the list
            comments.extend(data.get('data', []))

            # Check if there is a next page for pagination
            url = data.get('paging', {}).get('next', None)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching comments: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch comments: {str(e)}")

    # Sort comments by created_time (newest first)
    comments.sort(key=lambda x: x.get('created_time', ''), reverse=True)

    return comments

def post_facebook_reply(access_token: str, full_comment_id: str, reply_text: str):
    """Post AI-generated reply to a Facebook comment."""
    url = f"https://graph.facebook.com/v22.0/{full_comment_id}/comments"
    params = {"access_token": access_token, "message": reply_text}

    try:
        response = requests.post(url, params=params)
        response.raise_for_status()
        logger.info(f"Reply posted successfully: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error posting reply: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to post reply: {str(e)}")

def process_comments(config: FacebookConfig, gemini_api_key: str, sheet_id: str, credentials_dict: dict, sheet_name: str):
    """Fetch posts, get comments, generate replies, and post to Facebook."""
    global is_running
    try:
        is_running = True  # Mark as running

        existing_replies = load_existing_replies(sheet_id, credentials_dict, sheet_name)
        posts = get_facebook_posts(config.page_id, config.access_token)

        for post in posts:
            if not is_running:  # Check if we should stop processing
                logger.info("Stopping process_comments early")
                return

            post_id = post['id']
            post_message = post.get('message', 'No post content')
            post_permalink_url = post.get('permalink_url', 'No URL')  # Get post permalink URL
            post_time = post.get('created_time', 'Unknown')  # Get the post time
            comments = get_facebook_comments(post_id, config.access_token)

            comments.sort(key=lambda x: x.get('created_time', ''), reverse=True)

            for comment in comments:
                if not is_running:  # Check if we should stop processing
                    logger.info("Stopping process_comments early")
                    return

                raw_comment_id = comment['id']
                comment_message = comment.get('message', 'No comment message')
                comment_permalink_url = comment.get('permalink_url', 'No comment URL')  # Get comment permalink URL
                comment_time = comment.get('created_time', 'Unknown')

                # Extract the commenter's name
                commenter_name = comment.get('from', {}).get('name', 'Anonymous')  # Get the commenter's name

                # using @method to commenter
                commenter_id = comment.get('from', {}).get('id', '')  # Get the commenter's ID
                # Construct the profile link
                if commenter_id:
                    commenter_profile_link = f"https://www.facebook.com/profile.php?id={commenter_id}"
                else:
                    commenter_profile_link = commenter_name

                post_id_cleaned = post_id.split('_')[-1]
                comment_id_cleaned = raw_comment_id.split('_')[-1]

                full_comment_id = f"{config.page_id}_{post_id_cleaned}_{comment_id_cleaned}"

                if full_comment_id in existing_replies:
                   
                    continue  # Skip this comment if it's already been replied to
                
                preset_reply = preset_replies_check(comment['message'])

                reply_text = generate_ai_reply(
                        comment['message'],        # Comment content
                        gemini_api_key,            # API key for AI model
                        post_message,              # Post message for context
                        preset_reply,              # Preset reply if any
                        commenter_name,            # Pass the commenter's name
                        commenter_profile_link,    # Pass the commenter's profile link
                    )
                try: 

                    # # Now post the reply to Facebook
                    post_facebook_reply(config.access_token, full_comment_id, reply_text)

                    # Store reply and other details to Google Sheets before posting to Facebook
                    store_data_in_sheet({
                        "Post ID": post_id,
                        "Post Content": post_message,
                        "Post URL": post_permalink_url,  # Added post URL
                        "Post Time": post_time,  # Added post time
                        "Comment ID": full_comment_id,
                        "Comment Content": comment_message,  # Added comment content
                        "Comment URL": comment_permalink_url,  # Added comment URL
                        "Comment Time": comment_time,  # Added comment time
                        "Commenter Name": commenter_name,
                        "Reply": reply_text,  # Added reply text
                    }, sheet_id, credentials_dict, sheet_name)
                except Exception as e:
                    logger.error(f"Failed to post reply or store data: {e}")

              
    except Exception as e:
        logger.error(f"Error in scheduled task: {e}")
    finally:
        is_running = False  # Reset the flag when the job finishes

@app.post("/start-scheduler")
def start_scheduler(request: AIReplyRequest):
    """Starts a scheduler to fetch and reply to comments immediately and then at a user-defined interval."""
    global current_job

    if current_job:
        scheduler.remove_job("fetch_and_reply")

    # Define the first immediate run (using "now" to trigger immediately)
    current_job = scheduler.add_job(
        process_comments,
        IntervalTrigger(seconds=request.interval_seconds),
        id="fetch_and_reply",
        args=[request.config, request.gemini_api_key, request.google_sheet_id, 
              request.google_credentials.credentials, request.google_credentials.sheet_name],
        replace_existing=True,
        max_instances=1
    )

    # Trigger the first run immediately
    process_comments(
        request.config, 
        request.gemini_api_key, 
        request.google_sheet_id, 
        request.google_credentials.credentials, 
        request.google_credentials.sheet_name
    )  # Call process_comments immediately

    # Start the scheduler if it's not already running
    if not scheduler.running:
        scheduler.start()

    return {
        "status": "Scheduler started immediately and will repeat at the interval", 
        "interval": request.interval_seconds,
        "sheet_link": f"https://docs.google.com/spreadsheets/d/{request.google_sheet_id}/edit",
        "sheet_name": request.google_credentials.sheet_name
    }

@app.post("/stop-scheduler")
def stop_scheduler():
    """Stops the scheduled task and halts running jobs."""
    global is_running, current_job

    if current_job:
        # Stop the job and set the flag to False to stop the current running task
        is_running = False
        scheduler.remove_job("fetch_and_reply")
        current_job = None
        return {"status": "Scheduler stopped, and any running task will stop."}

    return {"status": "No active scheduler to stop"}
