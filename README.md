# Therapist.io

An ML and AI powered speech therapist web application that provides empathetic responses to user texts or speeches.

## Features

- **Speech or text-based user input**
- **Responses generated via LLM (Gemini API)**
- **Sentiment analysis of user responses and chat history for nuanced feedback**
- **Fallback responses for when LLM is unavailable**
- **Simple, modern UI**

## Tech Stack

- **Backend**: Flask 
- **AI/ML**: HuggingFace Transformers, Google Gemini API
- **Frontend**: HTML, CSS, JavaScript
- **Speech**: Web Speech API 

## Installation

1. **Clone the Repository**

   git clone <your-repo-url>
   cd AI-Speech-Therapist

2. **Install Dependencies**

   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. **Set up Environment Variables**
   - Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the Application**
 
   python app.py

   Navigate to `http://localhost:5000` in your browser.

## Project Structure

```
AI-Speech-Therapist/
├── app.py                  # Backend
├── utils.py                # Sentiment analysis, prompt builder, Gemini integration
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── static/                 # Static assets
│   └── therapist.png       # Therapist image
└── templates/              # HTML templates
    └── index.html          # Frontend interface
```

## How It Works

1. **User Input**: User types or speaks a message
2. **Sentiment Analysis**: Each message is analyzed for emotional content
3. **Context Building**: Chat history and sentiment data are combined
4. **AI Response**: Gemini AI generates an empathetic response
5. **Speech Output**: Response is displayed and spoken aloud


## Troubleshooting

**"PyTorch not found" error:**
- Ensure you're using Python 3.11 or lower (PyTorch doesn't support Python 3.13)

**Speech recognition not working:**
- Use HTTPS in production (required for speech API)
- Ensure browser supports Web Speech API

**Gemini API errors:**
- Check your API key is correct
- Verify you have sufficient API quota
- Check internet connectivity

## License

MIT License. Feel free to use, modify, and distribute.

⚠️ Disclaimer

This project is for educational and demonstrative purposes only.

It is **not** intended to diagnose, treat, or provide professional mental health advice. Please do not use this application as a substitute for therapy or mental health care. If you are experiencing distress or need help, contact a licensed professional or a crisis line.

