from fastapi import FastAPI, HTTPException, Request, Header, Body, Depends, Path, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime, timedelta, date
from math import exp, pow
from mangum import Mangum
import boto3
import uuid
import random
from boto3.dynamodb.conditions import Attr, Key
from decimal import Decimal
import os
import pysqlite3 as sqlite3
from pydantic import BaseModel, Field, constr
from starlette.config import Config
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse, HTMLResponse
from authlib.integrations.starlette_client import OAuthError
from typing import List, Optional
import hashlib
import csv
import tempfile
import io



# Environment variables
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID') or None
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET') or None
SECRET_KEY = os.environ.get('SECRET_KEY') or None
EMAIL_ALLOWLIST = os.environ.get('EMAIL_ALLOWLIST', '').split(',') # Will default to []

# FastAPI Boilerplate
app = FastAPI()

# Configure SessionMiddleware for OAuth
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Initialize DynamoDB 
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
flashcards_table = dynamodb.Table('flashcard_data')
histories_table = dynamodb.Table('flashcard_histories')
users_table = dynamodb.Table('flashcard_users')

if GOOGLE_CLIENT_ID is None or GOOGLE_CLIENT_SECRET is None:
    raise BaseException('Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in env vars')

config_data = {'GOOGLE_CLIENT_ID': GOOGLE_CLIENT_ID, 'GOOGLE_CLIENT_SECRET': GOOGLE_CLIENT_SECRET}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'prompt': 'select_account'
        },
)

# Math constants
w = [0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61]
FACTOR = 19/81
R = 0.9  # Desired retention rate
DECAY = -0.5

@app.route('/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth')  # This creates the url for the /auth endpoint
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.route('/auth')
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        return HTMLResponse(f'<p>Login unsuccessful: {str(e)} </p>')
    
    user_data = await oauth.google.userinfo(token=token)
    request.session['user'] = dict(user_data)

    # Fetch user email
    user_email = user_data.get('email')
    
    # Check if the user's email is in the EMAIL_ALLOWLIST
    if user_email is None or user_email not in EMAIL_ALLOWLIST:
        await logout(request)
        return HTMLResponse(f'<p>Login unsuccessful: {user_email} is not authorized to access this application.</p>', status_code=403)
    
    
    # Return an HTML response with a script to close the window
    return HTMLResponse('''
        <p>Login successful. You can now close this window or it will close automatically.</p>
    ''')


@app.route('/logout')
async def logout(request: Request):
    try:
        request.session.pop('user', None)
        return JSONResponse(content={"message": "Logout successful."}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"Logout failed: {str(e)}"}, status_code=500)


@app.get('/auth-info')
async def auth_info(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return user


def fetch_user_id(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_id = user.get('sub')
    if not user_id:
        raise HTTPException(status_code=500, detail="User ID not found. Something's wrong with OAuth.")
    
    return user_id


@app.get("/validate-authentication")
def validate_authentication(user_id: str = Depends(fetch_user_id)):
    return {"message": "Authentication valid", "user_id": user_id}


def sha256_hash(s):
    return hashlib.sha256(s.encode('utf-8', errors='replace')).hexdigest()


# Deck request body parameter for non-GET paths; falls back to "default" if not specified
def deck_request_body(user_id: str = Depends(fetch_user_id), deck: Optional[str] = Body(default=None)):
    if deck and deck not in get_userdata(user_id)["data"]["decks"]:
        raise HTTPException(status_code=400, detail=f"Deck '{deck}' does not exist.")
    return deck or "default"


# Deck query parameter for GET paths; allows None to signify 'all decks'
def deck_query_parameter(user_id: str = Depends(fetch_user_id), deck: Optional[str] = Query(None)):
    if deck and deck not in get_userdata(user_id)["data"]["decks"]:
        raise HTTPException(status_code=400, detail=f"Deck '{deck}' does not exist.")
    return deck


@app.post("/add")
def add_flashcard(card_front: str = Body(...), card_back: str = Body(...), 
                  deck: str = Depends(deck_request_body), user_id: str = Depends(fetch_user_id)):
    card_id = f"{sha256_hash(deck)}-{str(uuid.uuid4())}"
    # Store the flashcard details
    try:
        flashcards_table.put_item(
            Item={
                'card_id': card_id,
                'user_id': user_id,
                'card_front': card_front,
                'card_back': card_back
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add flashcard: {str(e)}")
    
    return {"message": "Flashcard added successfully", "card_id": card_id, "card_front": card_front, "card_back": card_back}


@app.delete("/delete/{card_id}")
def delete_flashcard(card_id: str = Path(..., title="The ID of the flashcard to delete"), user_id: str = Depends(fetch_user_id)):
    # Check if the flashcard exists and belongs to the user
    response = flashcards_table.get_item(Key={'user_id': user_id, 'card_id': card_id})
    if 'Item' not in response:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    if response['Item']['user_id'] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this flashcard")
    
    # Delete flashcard
    try:
        flashcards_table.delete_item(Key={'user_id': user_id, 'card_id': card_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete flashcard: {str(e)}")
    
    return {"message": "Flashcard deleted successfully"}


@app.post("/review/{card_id}")
async def review_flashcard(card_id: str = Path(..., title="The ID of the flashcard to review"), 
                           grade: int = Body(..., embed=True), 
                           user_id: str = Depends(fetch_user_id)):
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('flashcard_data')

    # Attempt to fetch the flashcard
    response = table.get_item(Key={'user_id': user_id, 'card_id': card_id})
    if 'Item' not in response:
        raise HTTPException(status_code=404, detail="Card not found")

    card = response['Item']

    # Can't review cards before review_date
    today = datetime.utcnow().date()
    if 'review_date' in card and today < datetime.strptime(card['review_date'], '%Y-%m-%d').date():
        raise HTTPException(status_code=403, detail=f"Card {card_id} is not due for review yet.") 

    if grade not in [1, 2, 3, 4]:
        raise ValueError("Grade must be between 1 and 4.")

    ### FSRS v4.5 Logic below ###

    # Helper function to calculate initial difficulty
    def D0(G):
        return w[4] - (G - 3) * w[5]

    # Difficulty
    difficulty = float(card.get('difficulty', D0(grade)))

    if 'difficulty' in card: # If this is a subsequent review
        difficulty = (w[7] * D0(3) + (1 - w[7])) * (difficulty - w[6] * (grade - 3))
        difficulty = max(1, min(difficulty, 10))  # Ensure difficulty is within bounds


    # Stability
    def calculate_new_stability_on_success(D, S, G):
        inner_term = exp(w[8]) * (11 - D) * S**(-w[9]) * (exp(w[10] * (1 - R)) - 1)
        if G == 2: # "Hard" multiplies by .29 
            inner_term *= w[15]
        elif G == 4: # "Easy" multiplies by 2.61
            inner_term *= w[16]
        return S * (inner_term + 1)

    def calculate_new_stability_on_fail(D, S):
        return w[11] * pow(D, (-w[12])) * (pow((S + 1), w[13]) - 1) * exp(w[14] * (1 - R))

    stability = float(card.get('stability', w[grade - 1]))

    if 'stability' in card: # If this is a subsequent review
        if grade == 1: # Failure
            stability = calculate_new_stability_on_fail(difficulty, stability)
        else: # Success
            stability = calculate_new_stability_on_success(difficulty, stability, grade)

    # Calculate next review date
    I = (stability / FACTOR) * (pow(R, 1 / DECAY) - 1)
    next_review_date = datetime.utcnow().date() + timedelta(days=int(I))


    # Update flashcard_data with results of review
    try:
        table.update_item(
            Key={'user_id': user_id, 'card_id': card_id},
            UpdateExpression="set difficulty = :d, stability = :s, review_date = :r, last_review_date = :l",
            ExpressionAttributeValues={
                ':d': Decimal(str(difficulty)),
                ':s': Decimal(str(stability)),
                ':r': next_review_date.strftime('%Y-%m-%d'),
                ':l': datetime.utcnow().date().strftime('%Y-%m-%d')
            },
        )
    except Exception as e:
        # Handle potential errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


    # Update flashcard_histories
    today_str = datetime.utcnow().date().strftime('%Y-%m-%d')
    try:
        response = histories_table.update_item(
            Key={
                'user_id': user_id,
                'date': today_str
            },
            UpdateExpression="SET all_reviews = if_not_exists(all_reviews, :start) + :inc, "
                             "new_reviews = if_not_exists(new_reviews, :start) + :new_inc",
            ExpressionAttributeValues={
                ':inc': Decimal('1'),
                ':new_inc': Decimal('0') if 'review_date' in card else Decimal('1'), # If it s a new card, increment new_reviews
                ':start': Decimal('0')
            },
            ReturnValues="UPDATED_NEW"  # Returns the new values of the updated attributes
        )
        updated_values = response.get('Attributes', {})
        new_reviews = updated_values.get('new_reviews', 0)
        all_reviews = updated_values.get('all_reviews', 0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update flashcard history: {str(e)}")

    # Return all the review details
    return JSONResponse(content={
        "message": "Review updated successfully",
        "card_id": card_id,
        "user_id": user_id,
        "difficulty": difficulty,
        "stability": stability,
        "next_review_date": next_review_date.strftime('%Y-%m-%d'),
        "last_review_date": datetime.utcnow().date().strftime('%Y-%m-%d'),
        "new_reviews": str(new_reviews),
        "all_reviews": str(all_reviews)
    })


# Helper function to get list of flashcards from DynamoDB with optional deck and filter_expression
def fetch_flashcards(user_id: str, deck: str = None, filter_expression = None):
    try:
        flashcards = []

        # Construct the key condition expression
        key_condition_expression = Key('user_id').eq(user_id)
        if deck: # If deck isn't "" or None
            key_condition_expression &= Key('card_id').begins_with(f"{sha256_hash(deck)}-")

        query_kwargs = {
            "KeyConditionExpression": key_condition_expression
        }

        if filter_expression:
            query_kwargs["FilterExpression"] = filter_expression

                # Start the query and pagination loop
        response = flashcards_table.query(**query_kwargs)
        flashcards.extend(response.get('Items', []))

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            query_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = flashcards_table.query(**query_kwargs)
            flashcards.extend(response.get('Items', []))

        return flashcards
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/list")
def list_flashcards(user_id: str = Depends(fetch_user_id), 
                    deck: str = Depends(deck_query_parameter)):
    try:
        flashcards = fetch_flashcards(user_id, deck)

        if flashcards:
            return {"flashcards": flashcards}
        else:
            return {"message": "No flashcards found."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/next")
def get_next_card(user_id: str = Depends(fetch_user_id), deck: str = Depends(deck_query_parameter)):
    today = datetime.utcnow().date().strftime('%Y-%m-%d')
    
    try:
        # Fetch user's preferred max_new_cards and deck
        userdata = get_userdata(user_id)
        max_new_cards = int(userdata.get('max_new_cards', 30))  # Default to 30 if not set
        
        history = histories_table.get_item(Key={'date': today, 'user_id': user_id}).get('Item', {})
        new_reviews_today = int(history.get('new_reviews', 0))  # Default to 0 if not set
        new_cards_remaining = max(max_new_cards - new_reviews_today, 0)

        # Want to fetch due cards (review_date before today) and new cards (cards with no review_date)
        filter_expression = Attr('review_date').lte(today) | Attr('review_date').not_exists()

        unfiltered_cards = fetch_flashcards(user_id, deck, filter_expression)

        # Filter to "new cards" and "review cards", pick one randomly
        new_cards = [card for card in unfiltered_cards if 'review_date' not in card][:new_cards_remaining]

        # Find the cards with earliest review date (most overdue cards)
        review_dates = [card['review_date'] for card in unfiltered_cards if 'review_date' in card]
        target_date = min(review_dates) if review_dates else None
        review_cards = [card for card in unfiltered_cards if card.get('review_date') == target_date] if target_date else []
        
        combined_subset = new_cards + review_cards
        
        if combined_subset:
            selected_card = random.choice(combined_subset)
            return { "flashcard": selected_card }
        else:
            return {"message": "No cards to review right now."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching the next card.")


@app.put("/edit/{card_id}")
def edit_flashcard(card_id: str = Path(..., title="The ID of the flashcard to edit"),
                   card_front: str = Body(None), card_back: str = Body(None),
                   user_id: str = Depends(fetch_user_id)):
    # Check if the flashcard exists
    response = flashcards_table.get_item(Key={'user_id': user_id, 'card_id': card_id})
    if 'Item' not in response:
        raise HTTPException(status_code=404, detail="Card not found")

    # Check if data is empty
    if card_front is None or card_back is None:
        return {"message": "Did not find any specified 'card_front' or 'card_back'."}

    # Prepare the update expression without needing Expression Attribute Names
    update_expression = "set "
    expression_attribute_values = {}
    if card_front is not None:
        update_expression += "card_front = :card_front, "
        expression_attribute_values[':card_front'] = card_front
    if card_back is not None:
        update_expression += "card_back = :card_back, "
        expression_attribute_values[':card_back'] = card_back
    # Remove trailing comma and space
    update_expression = update_expression.rstrip(', ')

    # Update the flashcard in DynamoDB
    try:
        flashcards_table.update_item(
            Key={'user_id': user_id, 'card_id': card_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch the updated flashcard to return it
    updated_response = flashcards_table.get_item(Key={'user_id': user_id, 'card_id': card_id})
    if 'Item' not in updated_response:
        raise HTTPException(status_code=404, detail="Failed to fetch updated flashcard")

    # Return the updated flashcard in the response
    return {"message": "Flashcard updated successfully!", "flashcard": updated_response['Item']}


@app.put("/create-deck/{deck}")
def create_deck(deck: str = Path(..., title="The name of the deck to create"), user_id: str = Depends(fetch_user_id)):
    try:
        decks = get_userdata(user_id).get("data", {}).get("decks", [])
        updated_decks = decks + [deck] if deck not in decks else decks

        update_data = { 'decks': updated_decks }
        update_userdata(UserData(**update_data), user_id)

        return {"message": f"Deck '{deck}' created successfully."}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating deck.")


@app.delete("/delete-deck/{deck}")
def delete_deck(deck: str = Path(..., title="The name of the deck to delete"), user_id: str = Depends(fetch_user_id)):
    # Scan DynamoDB to find all flashcards for the user and the specified deck
    response = flashcards_table.scan(
        FilterExpression=Attr('user_id').eq(user_id) & Attr('card_id').begins_with(f"{sha256_hash(deck)}-")
    )

    flashcards = response.get('Items', [])

    # Delete each flashcard
    with flashcards_table.batch_writer() as batch:
        for card in flashcards:
            batch.delete_item(
                Key={
                    'user_id': user_id,
                    'card_id': card['card_id']
                }
            )

    # Get the user's current data
    user_data = get_userdata(user_id)

    # Remove the deck from the user's list of decks
    updated_decks = [d for d in user_data['data']['decks'] if d != deck]

    # Update the user's data with the new list of decks
    update_data = {'decks': updated_decks}
    if user_data['data']['deck'] == deck:
        update_data['deck'] = None

    try:
        update_userdata(UserData(**update_data), user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {str(e)}")

    return {"message": f"Deck '{deck}' and associated cards deleted successfully."}


@app.put("/rename-deck")
def rename_deck(old_deck_name: str = Body(...), new_deck_name: str = Body(...), user_id: str = Depends(fetch_user_id)):
    try:
        if not old_deck_name or not new_deck_name:
            raise HTTPException(status_code=400, detail="old_deck_name and new_deck_name must not be blank")

        # Get the user's current data
        user_data = get_userdata(user_id)
        
        # Check if the old deck exists in the user's list of decks
        if old_deck_name not in user_data['data']['decks']:
            raise HTTPException(status_code=404, detail=f"Deck '{old_deck_name}' not found in user's list of decks.")

        # Check if the new deck name is already in use
        if new_deck_name in user_data['data']['decks']:
            raise HTTPException(status_code=400, detail=f"Deck name '{new_deck_name}' is already in use.")
        
        old_hashed_deck = sha256_hash(old_deck_name)
        new_hashed_deck = sha256_hash(new_deck_name)

        # Update the user's list of decks
        updated_decks = [new_deck_name if deck == old_deck_name else deck for deck in user_data['data']['decks']]

        # Update the user's active deck if it matches the old deck name
        update_data = {'decks': updated_decks}
        if user_data['data']['deck'] == old_deck_name:
            update_data['deck'] = new_deck_name

        update_userdata(UserData(**update_data), user_id)

        # Scan the table to retrieve all cards for the user and the specified old deck
        response = flashcards_table.scan(
            FilterExpression=Attr('user_id').eq(user_id) & Attr('card_id').begins_with(f"{old_hashed_deck}-")
        )
        cards = response.get('Items', [])

        # Batch write requests
        with flashcards_table.batch_writer() as batch:
            for card in cards:
                old_card_id = card['card_id']
                new_card_id = f"{new_hashed_deck}-" + old_card_id.split("-", 1)[1]

                # Create a new item with the updated card_id
                new_card = card.copy()
                new_card['card_id'] = new_card_id
                batch.put_item(Item=new_card)

                # Delete the old item
                batch.delete_item(Key={'user_id': user_id, 'card_id': old_card_id})

        return {"message": f"Deck '{old_deck_name}' renamed to '{new_deck_name}' successfully."}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error while creating deck download: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while downloading flashcards.")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Depends(fetch_user_id), deck: str = Depends(deck_request_body)):
    try:
        file_content = await file.read()

        if file.filename.endswith('.anki2'):
            extracted_cards = await extract_anki2(file_content)
        elif file.filename.endswith('.csv'):
            extracted_cards = await extract_csv(file_content)
        else:
            raise HTTPException(status_code=400, detail="Filetype not supported (Is it .anki2 or .csv)?")

        create_deck(deck, user_id)

        with flashcards_table.batch_writer() as batch:
            for card in extracted_cards:
                item = {
                    'user_id': user_id,
                    'card_id': f"{sha256_hash(deck)}-{str(uuid.uuid4())}",
                    'card_front': card['card_front'],
                    'card_back': card['card_back'],
                    **{k: Decimal(str(v)) if isinstance(v, float) else v for k, v in card.items() if file.filename.endswith('.csv')}
                }
                batch.put_item(Item=item)

        return {"message": f"Successfully imported {len(extracted_cards)} cards into {deck}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while importing deck: {e}")


async def extract_csv(file_content):
    cards = []
    csvfile = io.StringIO(file_content.decode('utf-8'))  # Convert bytes to string
    reader = csv.DictReader(csvfile)

    reader = csv.DictReader(csvfile)
    for row in reader:
        card_front = row['card_front'].replace('\\n', '\n').replace('""', '"')
        card_back = row['card_back'].replace('\\n', '\n').replace('""', '"')

        # Convert numeric values
        for key, value in row.items():
            if key not in ['card_front', 'card_back'] and value:
                try:
                    row[key] = float(value)
                    if row[key].is_integer():
                        row[key] = int(row[key])
                except ValueError:
                    pass

        cards.append({
            'card_front': card_front, 
            'card_back': card_back, 
            **{k:v for k,v in row.items() if k not in ['user_id', 'card_id', 'card_front', 'card_back']}})
    return cards

# Helper function for /upload path to extract cards
async def extract_anki2(file_content):
    # Create a temporary file to write the content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.anki2') as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

        # Connect to the Anki SQLite database
        conn = sqlite3.connect(tmp_file_path)
        cards = []
        try:
            cursor = conn.cursor()
            query = """
            SELECT cards.id, notes.flds
            FROM cards
            JOIN notes ON cards.nid = notes.id
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
                card_id, flds = row
                fields = flds.split('\x1f')  # Fields are separated by \x1f
                if len(fields) >= 2:  # Assumes cards have both front and back
                    cards.append({'card_front': fields[0], 'card_back': fields[1]})
        finally:
            conn.close()
    return cards


@app.get("/download")
def download_deck(user_id: str = Depends(fetch_user_id), deck: str = Depends(deck_query_parameter)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', encoding='utf-8', suffix='.csv')
    temp_file_path = temp_file.name
    try:
        response = list_flashcards(user_id, deck)
        flashcards = response.get("flashcards", [])

        # Derive list of columns
        cols = set()
        for flashcard in flashcards:
            cols.update(flashcard.keys())
        cols.discard('user_id') # Use discard() in case it's an empty list

        # Write to csv
        writer = csv.DictWriter(temp_file, fieldnames=list(cols), quoting=csv.QUOTE_NONNUMERIC)

        writer.writeheader()
        for flashcard in flashcards:
            # Update flashcard value to preserve newline characters and identify the data type
            sanitized_flashcards = {} 
            for key, value in flashcard.items():
                if key == 'user_id': # Don't include user_id
                    continue
                if isinstance(value, str): 
                    # Try to convert to number
                    try:
                        value = float(value)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:  # If it fails, keep as str
                        value = value.replace('\n', '\\n')
                    sanitized_flashcards[key] = value
                else: # If the value is not int, float or str, convert it to string format 
                    sanitized_flashcards[key] = str(value)

            writer.writerow(sanitized_flashcards)

        return FileResponse(temp_file_path, headers={'Content-Disposition': f'attachment; filename="{deck}.csv"'}, media_type='text/csv')

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred while downloading flashcards.")
    finally:
        temp_file.close()


# To add/remove fields, specify in UserData class
# They will get picked up dynamically by the PUT and GET /user-data paths

class BlockedSite(BaseModel):
    url: str
    active: bool = True

class UserData(BaseModel):
    max_new_cards: int = Field(
        ge=0,
        description="Maximum new reviews allowed per day",
        default=30
    )
    deck: Optional[str] = Field(
        default=None,
        description="Deck used for /next, /add, /list, /upload if otherwise unspecified"
    )
    decks: List[constr(strip_whitespace=True, min_length=1)] = Field(
        default=["default"],
        description="List of user's decks"
    )
    blocked_sites: List[BlockedSite] = Field(
        default_factory=lambda: [
            BlockedSite(url="https://www.reddit.com", active=True)
        ],
        description="List of blocked sites with accompanying flag for if they are enabled"
    )

# Setting userdata
@app.put("/user-data")
def update_userdata(user_data: UserData, user_id: str = Depends(fetch_user_id)):
    # Create a dictionary with only the fields that are explicitly provided in the request payload
    update_data = user_data.dict(exclude_unset=True)

    if not update_data:
        return {"message": "No fields to update"}

    # Dynamically build the update expression and attribute values
    update_expression = "set "
    expression_attribute_values = {}
    for field, value in update_data.items():
        update_expression += f"{field} = :{field}, "
        expression_attribute_values[f":{field}"] = value

    # Remove trailing comma and space from the update expression
    update_expression = update_expression.rstrip(", ")

    # Update in users_table
    try:
        response = users_table.update_item(
            Key={'user_id': user_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues="UPDATED_NEW"
        )
        return {"message": "User data updated successfully", "updatedAttributes": response.get('Attributes')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {str(e)}")


# Fetch preferences
@app.get("/user-data")
def get_userdata(user_id: str = Depends(fetch_user_id)):
    try:
        # Return userdata, providing default values where missing
        response = users_table.get_item(Key={'user_id': user_id})
        user_data = response.get('Item', {})  

        user_data.pop('user_id', None) # Remove redundant user_id
        merged_user_data = UserData.parse_obj(user_data)
        return {"data": merged_user_data.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user data: {str(e)}")



# Adapter for AWS Lambda
handler = Mangum(app)
