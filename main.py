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
from datetime import datetime, timedelta
import gspread
from google.oauth2 import service_account
import json
import tempfile
from dotenv import load_dotenv

app = FastAPI(title="Facebook Comment Reply Bot with Scheduler")

# Load environment variables from .env file
load_dotenv()

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Global variables
current_job = None
is_running = False
job_start_time = None  # Track when the job was started
preset_replies = {
    "hi there": "Hi there too!",
    "not so good": "We're sorry to hear that. Could you let us know what we can improve?",
    "how are you": "I'm doing well! How about you?",
    "rambunctious dinosaur": "That sounds like a wild dinosaur!",
    "শুভ কামনা রইলো": "Thank you!"
}
additional_instructions = ""

class FacebookConfig(BaseModel):
    page_id: str
    access_token: str

class GoogleCredentials(BaseModel):
    credentials: dict
    sheet_name: str = "Sheet1"

class AIReplyRequest(BaseModel):
    config: FacebookConfig
    interval_seconds: int
    google_sheet_id: str
    google_credentials: GoogleCredentials
    stop_time_after: Optional[int] = None  # New field for auto-stop

class AdditionalPromptConfig(BaseModel):
    additional_instructions: str

def get_sheets_client(credentials_dict, temp_file=None):
    try:
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
    temp_file = None
    try:
        client, temp_file = get_sheets_client(credentials_dict)
        try:
            sheet = client.open_by_key(sheet_id).worksheet(sheet_name)
            values = sheet.get_all_values()
            if len(values) > 1:
                comment_id_idx = values[0].index("Comment ID")
                return set(row[comment_id_idx] for row in values[1:])
            return set()
        except gspread.exceptions.WorksheetNotFound:
            spreadsheet = client.open_by_key(sheet_id)
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1, cols=11)
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
    temp_file = None
    try:
        client, temp_file = get_sheets_client(credentials_dict)
        spreadsheet = client.open_by_key(sheet_id)
        
        try:
            sheet = spreadsheet.worksheet(sheet_name)
            # Check if headers exist (first row is empty or doesn't match)
            existing_headers = sheet.row_values(1)
            expected_headers = ["Post ID", "Post Content", "Post URL", "Post Time", "Comment ID", 
                              "Comment Content", "Comment URL", "Comment Time", "Commenter Name", "Reply"]
            
            if not existing_headers or existing_headers != expected_headers:
                sheet.insert_row(expected_headers, index=1)
                logger.info(f"Added headers to existing sheet '{sheet_name}'")
                
        except gspread.exceptions.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1, cols=11)
            header = ["Post ID", "Post Content", "Post URL", "Post Time", "Comment ID", 
                     "Comment Content", "Comment URL", "Comment Time", "Commenter Name", "Reply"]
            sheet.update([header])
            logger.info(f"Created new sheet '{sheet_name}' with headers")

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
    global preset_replies
    comment = re.sub(r"[^\w\s]", "", comment).lower()
    comment_words = set(comment.split())

    for keyword, reply in preset_replies.items():
        clean_keyword = re.sub(r"[^\w\s]", "", keyword).lower()
        keyword_words = set(clean_keyword.split())
        if len(keyword_words.intersection(comment_words)) / len(keyword_words) >= 0.75:
            return reply
    return None

@app.post("/set-additional-instructions")
async def set_additional_instructions(prompt_config: AdditionalPromptConfig):
    global additional_instructions
    additional_instructions = prompt_config.additional_instructions
    return {"status": "success", "message": "Additional instructions updated"}

@app.get("/get-additional-instructions")
async def get_additional_instructions():
    return {"additional_instructions": additional_instructions}

def generate_ai_reply(comment: str, post_message: Optional[str] = None, 
                     preset_reply: Optional[str] = None, commenter_name: Optional[str] = None, 
                     commenter_profile_link: Optional[str] = None) -> str:
    try:
        time.sleep(1)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API key not found in environment variables")
            
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        is_bengali = any('\u0980' <= char <= '\u09FF' for char in comment)

        if is_bengali:
            comment_language="bengali"
        else:
            comment_language='others'

        prompt = f"""
        You are a page moderator.
        Post content: {post_message}
        Comment: {comment}
        Comment language:{comment_language}
        Commenter Profile Link: {commenter_profile_link}
        Preset Reply (if any): {preset_reply}

        if preset reply is none then ignore it mostly.
        Generate a polite reply that:
        2. Enhance the preset reply if available. If no preset reply exists, generate a relevant response based on the comment and post content."
        3. Is professional and friendly.
        4. Is concise (max 200 characters).
        5. Does not express uncertainty.
        6. If the comment is in Bengali, reply in Bengali. If not, reply in English.
        """

        if additional_instructions:
            prompt += f"\n\nAdditional Instructions:\n{additional_instructions}"

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"AI reply generation failed: {e}")
        return "Thank you for your comment"

@app.post("/add-preset-reply")
def add_preset_reply(preset_data: dict):
    if not preset_data:
        raise HTTPException(status_code=400, detail="No preset data provided")

    global preset_replies
    updated_keys = []
    new_keys = []

    for key, reply in preset_data.items():
        if not key or not reply:
            raise HTTPException(status_code=400, detail="Empty key or reply detected")
            
        if key in preset_replies:
            updated_keys.append(key)
        else:
            new_keys.append(key)
            
        preset_replies[key] = reply

    logger.info(f"Added {len(new_keys)} new preset replies and updated {len(updated_keys)} existing ones")
    return {
        "status": "Success",
        "added": new_keys,
        "updated": updated_keys,
        "total_presets": len(preset_replies)
    }

@app.get("/get-sheet-link/{sheet_id}")
async def get_sheet_link(sheet_id: str):
    return {"sheet_link": f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"}

def get_facebook_posts(page_id: str, access_token: str) -> List[Dict[str, Any]]:
    url = f"https://graph.facebook.com/v22.0/{page_id}/feed"
    params = {"access_token": access_token, "fields": "id,message,created_time,permalink_url"}
    posts = []
    
    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            posts.extend(data.get('data', []))
            url = data.get('paging', {}).get('next', None)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching posts: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch posts: {str(e)}")

    posts.sort(key=lambda x: x.get('created_time', ''), reverse=True)
    return posts

def get_facebook_comments(post_id: str, access_token: str) -> List[Dict[str, Any]]:
    url = f"https://graph.facebook.com/v22.0/{post_id}/comments"
    params = {"access_token": access_token, "fields": "id,message,created_time,permalink_url,from"}
    comments = []
    
    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            comments.extend(data.get('data', []))
            url = data.get('paging', {}).get('next', None)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching comments: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch comments: {str(e)}")

    comments.sort(key=lambda x: x.get('created_time', ''), reverse=True)
    return comments

def post_facebook_reply(access_token: str, full_comment_id: str, reply_text: str):
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
    
def get_page_name(page_id, access_token):
    url = f"https://graph.facebook.com/v12.0/{page_id}"
    params = {'fields': 'name', 'access_token': access_token}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('name', '[Unknown]')
    else:
        print(f"Error fetching page name: {response.json()}")
        return None

def get_replier_names(full_comment_id, access_token):
    url = f"https://graph.facebook.com/v12.0/{full_comment_id}/comments"
    params = {'fields': 'from', 'access_token': access_token}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        replies = response.json().get('data', [])
        if replies:
            return [reply.get('from', {}).get('name', '[Unknown]') for reply in replies]
        return []
    else:
        print(f"Error fetching replies: {response.status_code} - {response.json()}")
        return 'error'

def process_comments(config: FacebookConfig, sheet_id: str, credentials_dict: dict, sheet_name: str, stop_time_after: Optional[int] = None):
    global is_running
    
    try:

        is_running = True
        # Set a local job_start_time for this execution only
        local_job_start_time = datetime.now()
        
        logger.info(f"Starting job execution at {local_job_start_time}")
        
        posts = get_facebook_posts(config.page_id, config.access_token)
        page_name = get_page_name(config.page_id, config.access_token)

        for post in posts:
            # Check if we should stop based on the time elapsed for this execution
            if stop_time_after and datetime.now() >= local_job_start_time + timedelta(seconds=stop_time_after):
                print('time has elapsed ')
                logger.info(f"Stopping this job execution due to stop_time_after ({stop_time_after} seconds)")
                return
                
            if not is_running:
                logger.info("Stopping process_comments early due to manual stop")
                return

            post_id = post['id']
            post_message = post.get('message', 'No post content')
            post_permalink_url = post.get('permalink_url', 'No URL')
            post_time = post.get('created_time', 'Unknown')
            comments = get_facebook_comments(post_id, config.access_token)
            comments.sort(key=lambda x: x.get('created_time', ''), reverse=True)

            for comment in comments:
                # Check if we should stop based on the time elapsed for this execution
                if stop_time_after and datetime.now() >= local_job_start_time + timedelta(seconds=stop_time_after):
                    logger.info(f"Stopping this job execution due to stop_time_after ({stop_time_after} seconds)")
                    return
                    
                if not is_running:
                    logger.info("Stopping process_comments early due to manual stop")
                    return

                raw_comment_id = comment['id']
                comment_message = comment.get('message', 'No comment message')
                comment_permalink_url = comment.get('permalink_url', 'No comment URL')
                comment_time = comment.get('created_time', 'Unknown')
                commenter_name = comment.get('from', {}).get('name', 'Anonymous')
                commenter_id = comment.get('from', {}).get('id', '')
                commenter_profile_link = f"https://www.facebook.com/profile.php?id={commenter_id}" if commenter_id else commenter_name

                post_id_cleaned = post_id.split('_')[-1]
                comment_id_cleaned = raw_comment_id.split('_')[-1]
                full_comment_id = f"{config.page_id}_{post_id_cleaned}_{comment_id_cleaned}"
                comment_id_for_reply = raw_comment_id
                replier_names = get_replier_names(comment_id_for_reply, config.access_token)

                if page_name in replier_names or replier_names == 'error':
                    continue
                
                preset_reply = preset_replies_check(comment['message'])
                reply_text = generate_ai_reply(
                    comment['message'],
                    post_message,
                    preset_reply,
                    commenter_name,
                    commenter_profile_link
                )
                
                try: 
                    post_facebook_reply(config.access_token, full_comment_id, reply_text)
                    store_data_in_sheet({
                        "Post ID": post_id,
                        "Post Content": post_message,
                        "Post URL": post_permalink_url,
                        "Post Time": post_time,
                        "Comment ID": full_comment_id,
                        "Comment Content": comment_message,
                        "Comment URL": comment_permalink_url,
                        "Comment Time": comment_time,
                        "Commenter Name": commenter_name,
                        "Reply": reply_text,
                    }, sheet_id, credentials_dict, sheet_name)
                except Exception as e:
                    logger.error(f"Failed to post reply or store data: {e}")

    except Exception as e:
        logger.error(f"Error in scheduled task: {e}")
    finally:
        # We don't set is_running to False here to allow the job to be executed again at the next interval
        # We only set it to False when manually stopping the scheduler
        pass

@app.post("/start-scheduler")
def start_scheduler(request: AIReplyRequest):



    global current_job, is_running

    if current_job:
        scheduler.remove_job("fetch_and_reply")
        
    # Reset the running flag to allow new executions
    is_running = True
    stop_time_after = request.stop_time_after

    current_job = scheduler.add_job(
        process_comments,
        IntervalTrigger(seconds=request.interval_seconds),
        id="fetch_and_reply",
        args=[
            request.config, 
            request.google_sheet_id, 
            request.google_credentials.credentials, 
            request.google_credentials.sheet_name,
            stop_time_after
        ],
        replace_existing=True,
        max_instances=1
    )

    # Trigger first run immediately
    process_comments(
        request.config,
        request.google_sheet_id,
        request.google_credentials.credentials,
        request.google_credentials.sheet_name,
        stop_time_after
    )

    if not scheduler.running:
        print('scheduler isnt running bro')
        scheduler.start()

    return {
        "status": "Scheduler started. Each execution will run" + 
                 (f" for up to {stop_time_after} seconds" if stop_time_after else " until completion") +
                 f" and repeat every {request.interval_seconds} seconds.",
        "interval_seconds": request.interval_seconds,
        "stop_time_after_seconds": stop_time_after,
        "sheet_link": f"https://docs.google.com/spreadsheets/d/{request.google_sheet_id}/edit",
        "sheet_name": request.google_credentials.sheet_name
    }

@app.post("/stop-scheduler")
def stop_scheduler():

    
    global is_running, current_job

    if current_job:
        is_running = False
        scheduler.remove_job("fetch_and_reply")
        current_job = None
        return {"status": "Scheduler stopped manually."}
    return {"status": "No active scheduler to stop."}




