from fastapi import FastAPI, HTTPException, Header, Body, Depends, Path
from datetime import datetime, timedelta
from math import exp
from mangum import Mangum
import boto3
import uuid

# Initialize the FastAPI app
app = FastAPI()

# Initialize a DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
# Reference your DynamoDB tables
api_keys_table = dynamodb.Table('flashcard_api_keys')
flashcards_table = dynamodb.Table('flashcard_data')  

def fetch_user_id(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=400, detail="API key is required")
    response = api_keys_table.get_item(Key={'api_key': x_api_key})
    if 'Item' not in response:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return response['Item']['user_id']  # Assuming 'user_id' is a field in the item

@app.get("/validate-api-key/")
def validate_api_key(user_id: str = Depends(fetch_user_id)):
    return {"message": "API Key is valid", "user_id": user_id}

@app.post("/add/")
def add_flashcard(front: str = Body(...), back: str = Body(...), user_id: str = Depends(fetch_user_id)):    # Generate a unique card_id
    card_id = str(uuid.uuid4())
    
    # Store the flashcard details
    try:
        flashcards_table.put_item(
            Item={
                'card_id': card_id,
                'user_id': user_id,
                'front': front,
                'back': back
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


# Adapter for AWS Lambda
handler = Mangum(app)
