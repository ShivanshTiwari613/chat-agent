# Chat Agent

A lightweight AI agent with tool support and an optional E2B sandbox.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the required keys:

```env
GOOGLE_API_KEY=your_key_here
E2B_API_KEY=your_key_here
E2B_TEMPLATE_ID=
```

## Run

```bash
python main.py
```

## Notes

- Configure settings in `config/settings.py`.
- If using E2B, ensure the template is based on the code-interpreter runtime.
