from fastapi import FastAPI, HTTPException, Request, Header, Body, Depends, Path, File, UploadFile, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta, date
from math import exp, pow
from mangum import Mangum
import boto3
import uuid
import random
from boto3.dynamodb.conditions import Attr, Key
from decimal import Decimal
import shutil
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

# Returns the specified deck if provided, or the user's active deck, or throw exception
def get_deck(deck: str = Query(None), user_id: str = Depends(fetch_user_id)):
    fetched_deck = deck or users_table.get_item(Key={'user_id': user_id}).get('Item', {}).get('deck', None) 
    if fetched_deck is None:
        raise HTTPException(status_code=400, detail="No active deck found, please select one.")
    return fetched_deck


def sha256_hash(s):
    return hashlib.sha256(s.encode('utf-8', errors='replace')).hexdigest()


@app.get("/validate-authentication")
def validate_authentication(user_id: str = Depends(fetch_user_id)):
    return {"message": "Authentication valid", "user_id": user_id}


@app.post("/add")
def add_flashcard(card_front: str = Body(...), card_back: str = Body(...), deck: str = Depends(get_deck), user_id: str = Depends(fetch_user_id)):
    
    card_id = f"{sha256_hash(deck)}-{str(uuid.uuid4())}" # Generate card_id with deck and unique identifier
    
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
    
    return {"message": "Flashcard added successfully", "card_id": card_id, "user_id": user_id}


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


@app.get("/next")
def get_next_card(user_id: str = Depends(fetch_user_id), deck: str = Depends(get_deck)):
    today = datetime.utcnow().date().strftime('%Y-%m-%d')
    
    try:
        # Fetch user's preferred max_new_cards and deck
        prefs = users_table.get_item(Key={'user_id': user_id}).get('Item', {})
        max_new_cards = int(prefs.get('max_new_cards', 30))  # Default to 30 if not set
        
        history = histories_table.get_item(Key={'date': today, 'user_id': user_id}).get('Item', {})
        new_reviews_today = int(history.get('new_reviews', 0))  # Default to 0 if not set
        new_cards_remaining = max(max_new_cards - new_reviews_today, 0)

        # Initialize the list to hold all due cards
        unfiltered_cards = []

        # Start the query and pagination loop
        response = flashcards_table.query(
            KeyConditionExpression=Key('user_id').eq(user_id) & Key('card_id').begins_with(f"{sha256_hash(deck)}-"),
            FilterExpression=Attr('review_date').lte(today) | Attr('review_date').not_exists()
        )
        unfiltered_cards.extend(response.get('Items', []))

        # Handle pagination if there are more items to fetch
        while 'LastEvaluatedKey' in response:
            response = flashcards_table.query(
                KeyConditionExpression=Key('user_id').eq(user_id) & Key('card_id').begins_with(f"{sha256_hash(deck)}-"),
                FilterExpression=Attr('review_date').lte(today) | Attr('review_date').not_exists(),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            unfiltered_cards.extend(response.get('Items', []))

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
    if card_front is None and card_back is None:
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


@app.get("/list")
def list_flashcards(user_id: str = Depends(fetch_user_id), deck: str = Depends(get_deck)):
    try:
        flashcards = []

        # Start the query and pagination loop
        response = flashcards_table.query(
            KeyConditionExpression=Key('user_id').eq(user_id) & Key('card_id').begins_with(f"{sha256_hash(deck)}-")
        )
        flashcards.extend(response.get('Items', []))

        # Handle pagination if there are more items to fetch
        while 'LastEvaluatedKey' in response:
            response = flashcards_table.query(
                KeyConditionExpression=Key('user_id').eq(user_id) & Key('card_id').begins_with(f"{sha256_hash(deck)}-"),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            flashcards.extend(response.get('Items', []))

        if flashcards:
            return {"flashcards": flashcards}
        else:
            return {"message": "No flashcards found."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while listing the flashcards.")


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
def rename_deck(old_deck_name: str = Body(..., embed=True), new_deck_name: str = Body(..., embed=True), user_id: str = Depends(fetch_user_id)):
    old_hashed_deck = sha256_hash(old_deck_name)
    new_hashed_deck = sha256_hash(new_deck_name)

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

    # Get the user's current data
    user_data = get_userdata(user_id)

    # Check if the old deck exists in the user's list of decks
    if old_deck_name not in user_data['data']['decks']:
        raise HTTPException(status_code=404, detail=f"Deck '{old_deck_name}' not found in user's list of decks.")

    # Update the user's list of decks
    updated_decks = [new_deck_name if deck == old_deck_name else deck for deck in user_data['data']['decks']]

    # Update the user's active deck if it matches the old deck name
    update_data = {'decks': updated_decks}
    if user_data['data']['deck'] == old_deck_name:
        update_data['deck'] = new_deck_name

    try:
        update_userdata(UserData(**update_data), user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {str(e)}")

    return {"message": f"Deck '{old_deck_name}' renamed to '{new_deck_name}' successfully."}


# Helper function for /upload path to extract cards
async def extract_anki2(file_path):
    # Connect to the Anki SQLite database
    conn = sqlite3.connect(file_path)
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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Depends(fetch_user_id), deck: str = Depends(get_deck)):
    if not file.filename.endswith('.anki2'):
        return JSONResponse(status_code=400, content={"message": "This file type is not supported. Please upload an .anki2 file."})

    tmp_file_path = f"/tmp/{uuid.uuid4()}.anki2" 
    
    with open(tmp_file_path, 'wb') as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
    
    try:
        extracted_cards = await extract_anki2(tmp_file_path)

        with flashcards_table.batch_writer() as batch:
            for card in extracted_cards:
                batch.put_item(Item={
                    'user_id': user_id,
                    'card_id': f"{sha256_hash(deck)}-{str(uuid.uuid4())}",  # Generate a unique ID for each card
                    'card_front': card['card_front'],
                    'card_back': card['card_back'],
                })

        message = f"Successfully imported {len(extracted_cards)} cards."
    except Exception as e:
        message = f"An error occurred: {str(e)}"
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    
    return JSONResponse(content={"message": message})

# To add/remove fields, specify in UserData class
# They will get picked up dynamically by the PUT and GET /user-data paths

class UserData(BaseModel):
    max_new_cards: int = Field(ge=0, description="Maximum new reviews allowed per day", default=30)
    deck: Optional[str] = Field(default="default", description="Deck used for /next, /add, /list, /upload if otherwise unspecified")
    decks: List[constr(strip_whitespace=True, min_length=1)] = Field(default=["default"], description="List of user's decks")

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
        user_data = response.get('Item', {})  # Use get() to avoid KeyError

        user_data.pop('user_id', None) # Remove redundant user_id
        merged_user_data = UserData.parse_obj(user_data)
        return {"data": merged_user_data.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user data: {str(e)}")



# Adapter for AWS Lambda
handler = Mangum(app)
