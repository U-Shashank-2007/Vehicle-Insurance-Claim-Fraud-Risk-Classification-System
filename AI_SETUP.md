See [AI_FEATURES.md](AI_FEATURES.md) for the API configuration, privacy notes,
and responsible-use guidance.

Quick start (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-real-api-key"
python app.py
```

The AI investigation brief is optional. Without an API key or during a provider
outage, the original Random Forest prediction remains usable.
