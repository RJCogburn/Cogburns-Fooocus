"""Always-on block for prompts that combine sexual content with a minor.

There is no setting to turn this off. A prompt that only mentions a minor is allowed, but its
images are run through the NSFW check (black out NSFW) regardless of the UI setting.
"""
import json
import os
import re
import threading
import time

# Written by DiffusionDeck's Family Safety mode. While it exists, Black Out NSFW is forced on.
FAMILY_LOCK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'family_safety.lock')


def family_lock() -> bool:
    return os.path.exists(FAMILY_LOCK)


# Models DiffusionDeck's Family Safety has blocked, by file name: {"blocked": ["name.safetensors", ...]}
FAMILY_MODELS = os.path.join(os.path.dirname(FAMILY_LOCK), 'family_models.json')
FAMILY_ACTIVITY = os.path.join(os.path.dirname(FAMILY_LOCK), 'family_activity.jsonl')
# The parent's own blocked words and phrases: {"blocked": ["word", "two words", ...]}
FAMILY_WORDS = os.path.join(os.path.dirname(FAMILY_LOCK), 'family_words.json')
WORDS_MESSAGE = 'Blocked: this prompt uses a word that isn\'t allowed. Nothing was generated.'

# Cached policy files. The parent can change these at any time, so we re-read when the file's mtime
# changes rather than caching forever (and rather than reading on every check, which is slow).
_WORDS_CACHE: tuple[float, list] | None = None
_MODELS_CACHE: tuple[float, set] | None = None


def _cached_words() -> list:
    global _WORDS_CACHE
    try:
        mtime = os.stat(FAMILY_WORDS).st_mtime
    except OSError:
        _WORDS_CACHE = None
        return []
    if _WORDS_CACHE is not None and _WORDS_CACHE[0] == mtime:
        return _WORDS_CACHE[1]
    try:
        with open(FAMILY_WORDS, encoding='utf-8') as f:
            words = json.load(f).get('blocked', [])
    except (OSError, ValueError, AttributeError):
        words = []
    words = [w for w in (_normalize(str(w)) for w in words) if w]
    _WORDS_CACHE = (mtime, words)
    return words


def _cached_blocked_models() -> set:
    global _MODELS_CACHE
    try:
        mtime = os.stat(FAMILY_MODELS).st_mtime
    except OSError:
        _MODELS_CACHE = None
        return set()
    if _MODELS_CACHE is not None and _MODELS_CACHE[0] == mtime:
        return _MODELS_CACHE[1]
    try:
        with open(FAMILY_MODELS, encoding='utf-8') as f:
            blocked = {os.path.basename(n).casefold() for n in json.load(f).get('blocked', [])}
    except (OSError, ValueError, AttributeError):
        blocked = set()
    _MODELS_CACHE = (mtime, blocked)
    return blocked


def uses_blocked_word(*texts: str) -> bool:
    """True when a parent-blocked word or phrase appears as whole words (also in number-letter spelling)."""
    if not family_lock():
        return False
    words = _cached_words()
    if not words:
        return False
    text = ' '.join(_normalize(t or '') for t in texts)
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')(?:s|es)?\b')
    return bool(pattern.search(text) or pattern.search(text.translate(_LEET)))


def model_allowed(name) -> bool:
    if not family_lock() or not isinstance(name, str) or name == 'None':
        return True
    return os.path.basename(name).casefold() not in _cached_blocked_models()


def allowed_models(names: list) -> list:
    return [n for n in names if model_allowed(n)]


def check_model(name) -> None:
    if not model_allowed(name):
        raise PromptBlocked(f'Blocked: the model "{os.path.basename(name)}" isn\'t allowed by Family Safety. '
                            'Nothing was generated.')


# Pictures the NSFW check blacked out during the current request. Per worker thread, reset per request.
_local = threading.local()


def _blacked_out() -> int:
    return getattr(_local, 'blacked_out', 0)


def _set_blacked_out(value: int) -> None:
    _local.blacked_out = value


def count_censored(before, after) -> None:
    """Called with the pictures before and after the NSFW check; counts the ones it turned black."""
    pairs = zip(before, after) if isinstance(before, list) else [(before, after)]
    _set_blacked_out(_blacked_out() + sum(1 for b, a in pairs if a.max() == 0 and b.max() > 0))


def _block_kind(blocked: str) -> str:
    if not blocked:
        return ''
    if blocked == BLOCKED_MESSAGE:
        return 'minor'
    if blocked == WORDS_MESSAGE:
        return 'word'
    return 'model' if 'model' in blocked else 'other'


def log_activity(async_task, blocked: str = '') -> None:
    """Record what was asked for, so a parent can review it in DiffusionDeck."""
    count = _blacked_out()
    _set_blacked_out(0)
    if not family_lock():
        return
    entry = {
        'time': time.time(),
        'prompt': getattr(async_task, 'prompt', ''),
        'negative_prompt': getattr(async_task, 'negative_prompt', ''),
        'enhance_prompts': [t for c in getattr(async_task, 'enhance_ctrls', []) for t in c[:2] if isinstance(t, str) and t],
        'model': getattr(async_task, 'base_model_name', ''),
        'loras': [l[0] for l in getattr(async_task, 'loras', []) if l and l[0] != 'None'],
        'size': getattr(async_task, 'aspect_ratios_selection', '').split(' ')[0],
        'images': getattr(async_task, 'image_number', 0),
        'image_input': bool(getattr(async_task, 'input_image_checkbox', False)),
        'blocked': blocked,
        'kind': _block_kind(blocked),
        'blacked_out': count,
    }
    try:
        with open(FAMILY_ACTIVITY, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except OSError:
        pass

BLOCKED_MESSAGE = ('Blocked: this prompt combines sexual content with a minor. '
                   'Nothing was generated.')

class PromptBlocked(Exception):
    pass


_NUMBER_WORDS = ('one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|'
                 'fourteen|fifteen|sixteen|seventeen')
_UNDER_18 = r'(?:[1-9]|1[0-7]|' + _NUMBER_WORDS + r')'

_MINOR = re.compile(r'\b(?:' + '|'.join([
    r'child(?:ren|s)?', r'kids?', r'kiddos?', r'minors', r'under ?age(?:d)?', r'pre ?teens?', r'tweens?',
    r'toddlers?', r'infants?', r'newborns?', r'lolis?', r'lolicon', r'lolita', r'shotas?', r'shotacon',
    r'jailbait', r'(?:pre)?pubescent', r'school ?(?:girl|boy)s?', r'(?:little|young) (?:girl|boy)s?',
    r'teen(?:s|age|aged|ager|agers)?', r'adolescents?', r'youngsters?', r'juveniles?',
    r'(?:middle|elementary|grade|primary|junior high) school(?:er)?s?', r'kindergarten(?:er)?s?',
    _UNDER_18 + r' ?(?:years?|yrs?) ?old', _UNDER_18 + r' ?(?:yo|y o)',
    r'(?:aged?|age of) ' + _UNDER_18,
]) + r')\b')

_SEXUAL = re.compile(r'\b(?:' + '|'.join([
    r'nude', r'nudes', r'nudity', r'naked', r'nsfw', r'sex', r'sexy', r'sexual(?:ly|ized|ised)?', r'porn\w*',
    r'explicit', r'xxx', r'hentai', r'erotic\w*', r'lewd', r'vagina\w*', r'pussy', r'penis\w*', r'cocks?',
    r'dicks?', r'genital\w*', r'labia', r'clit\w*', r'anus', r'anal', r'breasts?', r'boobs?', r'tits?',
    r'nipples?', r'topless', r'bottomless', r'pubic', r'cum(?:s|shots?|ming)?', r'semen', r'sperm', r'ejaculat\w*',
    r'orgasm\w*', r'masturbat\w*', r'rap(?:e|ed|es|ing|ist)', r'molest\w*', r'intercourse', r'fellatio',
    r'blow ?jobs?', r'hand ?jobs?', r'cunnilingus', r'penetrat\w*', r'insertion', r'undress\w*', r'strip(?:s|ped|per|pers|ping|tease)?',
    r'spread legs', r'lingerie', r'panties', r'seductive', r'aroused', r'arousal', r'horny', r'fetish\w*',
    r'bdsm', r'bondage', r'gang ?bang\w*', r'creampie', r'impregnat\w*', r'inseminat\w*', r'ass up',
    r'grop\w*', r'fondl\w*',
]) + r')\b')

_LEET = str.maketrans({'3': 'e', '0': 'o', '4': 'a', '1': 'i', '5': 's', '@': 'a', '$': 's', '7': 't'})


def _normalize(text: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9@$]+', ' ', text.lower()).split())


def mentions_minor(*texts: str) -> bool:
    return any(_MINOR.search(n) or _MINOR.search(n.translate(_LEET)) for n in (_normalize(t or '') for t in texts))


def is_blocked(*texts: str) -> bool:
    """True when the texts together mention a minor and anything sexual."""
    text = ' '.join(_normalize(t or '') for t in texts)
    leet = text.translate(_LEET)
    if not (_MINOR.search(text) or _MINOR.search(leet)):
        return False
    return bool(_SEXUAL.search(text) or _SEXUAL.search(leet))


def check_tasks(async_task, tasks):
    """Raise PromptBlocked for a blocked prompt; force the NSFW check when a minor is mentioned.

    Run on the final prompts (after wildcards, styles and Fooocus V2 expansion)."""
    for t in tasks:
        texts = list(t['positive']) + [t.get('expansion', '')]
        if is_blocked(*texts):
            raise PromptBlocked(BLOCKED_MESSAGE)
        if uses_blocked_word(*texts):
            raise PromptBlocked(WORDS_MESSAGE)
        if mentions_minor(*texts):
            async_task.black_out_nsfw = True
