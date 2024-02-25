from fastapi import FastAPI, HTTPException, Header, Body, Depends, Path, File, UploadFile
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


# Initialize the FastAPI app
app = FastAPI()

# Initialize a DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
# Reference your DynamoDB tables
api_keys_table = dynamodb.Table('flashcard_api_keys')
flashcards_table = dynamodb.Table('flashcard_data')  
# Math constants
w = [0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61]
FACTOR = 19/81
R = 0.9  # Desired retention rate
DECAY = -0.5

def fetch_user_id(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=400, detail="API key is required")
    response = api_keys_table.get_item(Key={'api_key': x_api_key})
    if 'Item' not in response:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return response['Item']['user_id']  # Assuming 'user_id' is a field in the item

@app.get("/validate-api-key")
def validate_api_key(user_id: str = Depends(fetch_user_id)):
    return {"message": "API Key is valid", "user_id": user_id}

@app.post("/add")
def add_flashcard(card_front: str = Body(...), card_back: str = Body(...), user_id: str = Depends(fetch_user_id)):    # Generate a unique card_id
    card_id = str(uuid.uuid4())
    
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
    today = datetime.now().date()
    if 'review_date' in card and today < datetime.strptime(card['review_date'], '%Y-%m-%d').date():
        raise HTTPException(status_code=403, detail="This card is not due for review yet.") 

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
    next_review_date = datetime.now().date() + timedelta(days=int(I))

    try:
        table.update_item(
            Key={'user_id': user_id, 'card_id': card_id},
            UpdateExpression="set difficulty = :d, stability = :s, review_date = :r, last_review_date = :l",
            ExpressionAttributeValues={
                ':d': Decimal(str(difficulty)),
                ':s': Decimal(str(stability)),
                ':r': next_review_date.strftime('%Y-%m-%d'),
                ':l': datetime.now().date().strftime('%Y-%m-%d')
            },
        )
        # Construct and return a JSON response with the review details
        return JSONResponse(content={
            "message": "Review updated successfully",
            "card_id": card_id,
            "user_id": user_id,
            "difficulty": difficulty,
            "stability": stability,
            "next_review_date": next_review_date.strftime('%Y-%m-%d'),
            "last_review_date": datetime.now().date().strftime('%Y-%m-%d')
        })
    except Exception as e:
        # Handle potential errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/next")
def get_next_card(user_id: str = Depends(fetch_user_id)):
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('flashcard_data')  # Use your actual table name

    today = date.today().isoformat()

    # Scan for cards whose review date is today or in the past, including those without a review_date
    response = table.scan(
        FilterExpression=Attr('user_id').eq(user_id) & (Attr('review_date').lte(today) | Attr('review_date').not_exists())
    )

    due_cards = response.get('Items', [])
    if not due_cards:
        return {"message": "No cards to review right now."}

    # Filter cards with the earliest review date or no review date
    earliest_date = min((card.get('review_date', today) for card in due_cards), default=today)
    earliest_cards = [card for card in due_cards if card.get('review_date', today) == earliest_date]

    # Randomly select a card from the earliest review date (most overdue cards)
    selected_card = random.choice(earliest_cards)

    return selected_card

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

@app.delete("/clear")
def clear_flashcards(user_id: str = Depends(fetch_user_id)):
    # Scan DynamoDB to find all flashcards for the user
    response = flashcards_table.scan(
        FilterExpression=Attr('user_id').eq(user_id)
    )

    flashcards = response.get('Items', [])
    if not flashcards:
        return {"message": "No flashcards found for the user."}

    # Delete each flashcard
    with flashcards_table.batch_writer() as batch:
        for card in flashcards:
            batch.delete_item(
                Key={
                    'user_id': user_id,
                    'card_id': card['card_id']
                }
            )

    return {"message": "All flashcards cleared for the user."}


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
async def upload_file(file: UploadFile = File(...), user_id: str = Depends(fetch_user_id)):
    if not file.filename.endswith('.anki2'):
        return JSONResponse(status_code=400, content={"message": "This file type is not supported. Please upload an .anki2 file."})

    tmp_file_path = f"/tmp/{uuid.uuid4()}.anki2"  # Use uuid in case multiple people uploading
    
    with open(tmp_file_path, 'wb') as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
    
    try:
        extracted_cards = await extract_anki2(tmp_file_path)

        with flashcards_table.batch_writer() as batch:
            for card in extracted_cards:
                batch.put_item(Item={
                    'user_id': user_id,
                    'card_id': str(uuid.uuid4()),  # Generate a unique ID for each card
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


# Adapter for AWS Lambda
handler = Mangum(app)
