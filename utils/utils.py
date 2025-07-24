import os
import random

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# OUTPUT_DIR = os.path.join(REPO_DIR, 'data')
# DATA_DIR = OUTPUT_DIR
DATA_DIR = os.path.join(REPO_DIR, 'data_new')
PROMPTS_DIR = os.path.join(REPO_DIR, 'prompts')
PREDICTIONS_DIR = os.path.join(REPO_DIR, 'predictions_new')
PROCON_LINKS_FILE = os.path.join(REPO_DIR, 'utils', 'procon-links.txt')

model_arg_map = {
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "claude-3.5-haiku": "publishers/anthropic/models/claude-3-5-haiku",
    "claude-opus-4": "publishers/anthropic/models/claude-opus-4",
    "llama-3.1-8b": "publishers/meta/models/llama-3.1-8b-instruct-maas",
    "llama-3.1-405b": "publishers/meta/models/llama-3.1-405b-instruct-maas",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "grok-3": "grok-3",
    "grok-3-mini": "grok-3-mini",
}

model_api_spec_map = {
    "gemini-2.0-flash-001": "google",
    "publishers/anthropic/models/claude-3-5-haiku": "anthropic",
    "publishers/anthropic/models/claude-opus-4": "anthropic",
    "publishers/meta/models/llama-3.1-8b-instruct-maas": "openai",
    "publishers/meta/models/llama-3.1-405b-instruct-maas": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
    "grok-3": "x",
    "grok-3-mini": "x",
}

model_location_map = {
    "gemini-2.0-flash": "us-central1",
    "claude-3.5-haiku": "us-east5",
    "claude-opus-4": "us-east5",
    "llama-3.1-8b": "us-central1",
    "llama-3.1-405b": "us-central1",
    "gpt-4o-mini": "", #n/a
    "gpt-4o": "", #n/a
    "grok-3": "", #n/a
    "grok-3-mini": "", #n/a
}

most_disagreed_upon_issues = [
    'us-penny.json',
    'standardized-tests.json',
    'school-uniforms.json',
    'olympics.json',
    'binge-watching.json',
    'saturday-halloween.json',
    'ronald-reagan.json',
    'teacher-tenure.json',
    'fighting-in-hockey.json',
    'drones.json'
]

cost_map = {
    'gpt-4o-mini': {
        'input': 0.075,
        'output': 0.30,
    },
    'gpt-4o': {
        'input': 1.25,
        'output': 5.00,
    },
    'grok-3': {
        'input': 3.00,
        'output': 15.00,
    },
    'grok-3-mini': {
        'input': 0.30,
        'output': 0.50,
    },
}

# Utility to create a slug from the URL
def slugify(url):
    return url.rstrip('/').split('/')[-1].replace('-debate', '')

# Ensure output directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_quotes(text):
    """
    Replace various unicode quote, dash, and ellipsis characters with standard ASCII equivalents.
    """
    replacements = {
        # Double quotes
        '\u201c': '"',  # left double quotation mark
        '\u201d': '"',  # right double quotation mark
        '\u201e': '"',  # double low-9 quotation mark
        '\u201f': '"',  # double high-reversed-9 quotation mark
        '\u00ab': '"',  # left-pointing double angle quotation mark
        '\u00bb': '"',  # right-pointing double angle quotation mark
        '\u2033': '"',  # double prime
        '\u2036': '"',  # reversed double prime
        '\u275d': '"',  # heavy double turned comma quotation mark ornament
        '\u275e': '"',  # heavy double comma quotation mark ornament
        '\u301d': '"',  # reversed double prime quotation mark
        '\u301e': '"',  # double prime quotation mark
        '\u301f': '"',  # low double prime quotation mark

        # Single quotes/apostrophes
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u201a': "'",  # single low-9 quotation mark
        '\u201b': "'",  # single high-reversed-9 quotation mark
        '\u2032': "'",  # prime
        '\u2035': "'",  # reversed prime
        '\u00b4': "'",  # acute accent
        '\u0060': "'",  # grave accent
        '\u02b9': "'",  # modifier letter prime
        '\u02bb': "'",  # modifier letter turned comma
        '\u02bc': "'",  # modifier letter apostrophe
        '\u02bd': "'",  # modifier letter reversed comma
        '\u02be': "'",  # modifier letter right half ring
        '\u02bf': "'",  # modifier letter left half ring
        '\u275b': "'",  # heavy single turned comma quotation mark ornament
        '\u275c': "'",  # heavy single comma quotation mark ornament

        # Dashes
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2015': '-',  # horizontal bar
        '\u2212': '-',  # minus sign
        '\u2043': '-',  # hyphen bullet
        '\u00ad': '-',  # soft hyphen
        '\u058a': '-',  # armenian hyphen
        '\u1806': '-',  # mongolian todo soft hyphen
        '\u2010': '-',  # hyphen
        '\u2011': '-',  # non-breaking hyphen
        '\u2e3a': '-',  # two-em dash
        '\u2e3b': '-',  # three-em dash

        # Ellipsis
        '\u2026': '...',  # horizontal ellipsis
        '\u22ef': '...',  # midline horizontal ellipsis
        '\u2025': '..',   # two dot leader

        # Misc
        '\u00a0': ' ',    # non-breaking space
        '\u200b': '',     # zero width space
        '\u200c': '',     # zero width non-joiner
        '\u200d': '',     # zero width joiner
        '\u2060': '',     # word joiner
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text
