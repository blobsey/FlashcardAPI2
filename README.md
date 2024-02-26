# FlashcardAPIv2

Backend of [FlashcardExtension](https://github.com/blobsey/FlashcardExtension)

- Uses [FastAPI](https://fastapi.tiangolo.com/) framework 
- Uses [Mangum](https://mangum.io/) to be deployed to an AWS Lambda
- Supports file uploads of .anki2 files to import
- Uses AWS DynamoDB to store flashcard data, users, etc.
- Serves cards using [FSRSv4.5](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm) scheduling
- I think it's really cool :)