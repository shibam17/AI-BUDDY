
# AI Study Buddy

This project is a simple GenAI LLM application that uses Retrieval-Augmented Generation (RAG) to help students answer questions based on stored study materials.

## How It Works:
- The app retrieves relevant study materials from a vector database.
- It uses OpenAI's GPT-3 (or GPT-4) to generate responses based on these materials.

## Setup Instructions:

1. Clone the repository:

   ```bash
   git clone <repo-link>
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   - In `app.py`, replace `'your-openai-api-key'` with your OpenAI API key.

4. Run the application:
   ```bash
   python app.py
   ```

5. You can send POST requests to `http://localhost:5000/ask` with a JSON body like:
   ```json
   {
     "question": "What is photosynthesis?"
   }
   ```

6. The app will return a generated answer based on the context it retrieves from stored materials.
