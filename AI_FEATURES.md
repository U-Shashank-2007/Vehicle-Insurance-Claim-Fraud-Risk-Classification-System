# ClaimGuard AI Investigation Assistant

## What the AI feature does

ClaimGuard continues to make its numerical prediction with the existing
preprocessor and trained Random Forest. The optional AI layer is called only
after a prediction. It creates a plain-language investigation brief containing
the model result, neutral checks an investigator may perform, and a reminder
that human review is required.

The AI response does **not** change the score, threshold, priority, or
flagged-for-review state. It is not evidence, a fraud finding, legal advice, or
a recommendation to reject a claim.

## Configure the API

The integration uses an OpenAI-compatible Chat Completions API.

PowerShell (current terminal):
```powershell
$env:OPENAI_API_KEY="your-real-api-key"
$env:OPENAI_MODEL="gpt-4o-mini"
python app.py
```

Optional provider configuration:
```powershell
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

For another OpenAI-compatible service, set `OPENAI_BASE_URL` to its API base
and `OPENAI_MODEL` to a model supported by that provider. Never put a real
API key in source code, screenshots, Git history, or a committed `.env` file.

## Added capabilities

- AI-generated per-claim investigation brief.
- Neutral verification suggestions rather than fraud accusations.
- Identifier minimization before sending model features to the provider.
- API timeout/error handling; the original ML assessment remains available.
- Configurable OpenAI-compatible endpoint and model.
- Explicit separation between predictive score and generated explanation.

## Privacy and responsible use

The feature sends model input attributes and the already-computed risk score
to the configured third-party API. Common direct identifiers (names, email,
phone, address, customer/policy/claim IDs and date of birth) are excluded.
Do not submit real claimant data unless you have the appropriate permission,
legal basis, provider terms, retention controls and organizational approval.
A feature-value combination may still be identifying, so use synthetic or
properly de-identified data for demonstrations.

LLM output can be incorrect, incomplete or overconfident. Verify every
suggestion against original records. ClaimGuard is for educational and
investigator-support use; no claim should be automatically denied or labelled
fraudulent based on this software.
