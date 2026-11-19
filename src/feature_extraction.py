"""Deterministic URL-only feature extraction. No website is opened or contacted."""
import ipaddress
import math
from urllib.parse import urlparse

import tldextract

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "count_dots", "count_hyphens", "count_underscores", "count_slashes",
    "count_question_marks", "count_equal_signs", "count_at", "count_ampersands",
    "count_percent", "count_digits", "count_letters", "digit_ratio",
    "special_char_ratio", "url_entropy", "https", "has_ip_address", "has_port",
    "subdomain_count", "domain_length", "tld_length", "path_depth",
    "has_suspicious_word", "has_suspicious_extension", "has_shortener_domain",
    "has_hyphen_in_domain", "has_brand_name", "is_free_hosting",
    "has_multiple_subdomains",
]

SUSPICIOUS_WORDS = (
    "login", "verify", "account", "update", "secure", "banking",
    "confirm", "password", "signin", "wallet", "payment", "support",
    "sign-in", "log-in", "unlock", "recover", "authorize",
)

SUSPICIOUS_EXTENSIONS = (
    ".exe", ".bin", ".scr", ".vmp", ".dll", ".bat", ".cmd",
    ".msi", ".jar", ".vbs", ".ps1", ".hta",
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "cutt.ly", "rebrand.ly", "urlz.fr",
    "lihi.cc", "shorturl.at",
}

BRAND_NAMES = {
    "metamask", "coinbase", "kraken", "gemini", "uphold", "robinhood",
    "phantom", "ledger", "trezor", "binance", "telegram", "netflix",
    "instagram", "facebook", "apple", "microsoft", "att", "sbcglobal",
    "paypal", "chase", "bankofamerica", "wellsfargo", "dana", "orange",
    "whatsapp", "google", "amazon", "yahoo", "outlook", "office365",
    "icloud", "steam", "spotify", "tiktok",
}

FREE_HOSTING = {
    "godaddysites.com", "weebly.com", "weeblysite.com", "pages.dev",
    "vercel.app", "netlify.app", "web.app", "firebaseapp.com",
    "gitbook.io", "wixsite.com", "workers.dev", "r2.dev",
    "dweb.link", "surge.sh", "glitch.me", "replit.app",
    "square.site", "blogspot.com", "github.io", "ipfs.io",
    "fleek.co", "cloudwaysapps.com",
}

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=None)


def _normalise_url(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    if not value.lower().startswith(("http://", "https://")):
        value = "http://" + value
    return value


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def extract_url_features(url: str) -> dict:
    raw = str(url).strip()
    normalised = _normalise_url(raw)
    parsed = urlparse(normalised)
    hostname = (parsed.hostname or "").lower()
    ext = _EXTRACT(hostname) if hostname else None

    digits = sum(ch.isdigit() for ch in raw)
    letters = sum(ch.isalpha() for ch in raw)
    special = sum(not ch.isalnum() for ch in raw)

    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    domain = ext.domain if ext else ""
    subdomain = ext.subdomain if ext else ""
    suffix = ext.suffix if ext else ""
    registered_domain = ".".join(x for x in (domain, suffix) if x)

    try:
        port = 1 if parsed.port is not None else 0
    except ValueError:
        port = 0

    lower = raw.lower()
    lower_domain = domain.lower()
    subdomain_parts = [p for p in subdomain.split(".") if p]

    values = {
        "url_length": len(raw),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "count_dots": raw.count("."),
        "count_hyphens": raw.count("-"),
        "count_underscores": raw.count("_"),
        "count_slashes": parsed.path.count("/"),
        "count_question_marks": raw.count("?"),
        "count_equal_signs": raw.count("="),
        "count_at": raw.count("@"),
        "count_ampersands": raw.count("&"),
        "count_percent": raw.count("%"),
        "count_digits": digits,
        "count_letters": letters,
        "digit_ratio": digits / max(len(raw), 1),
        "special_char_ratio": special / max(len(raw), 1),
        "url_entropy": _entropy(raw),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": is_ip,
        "has_port": port,
        "subdomain_count": len(subdomain_parts),
        "domain_length": len(registered_domain),
        "tld_length": len(suffix),
        "path_depth": len([p for p in parsed.path.split("/") if p]),
        "has_suspicious_word": int(any(w in lower for w in SUSPICIOUS_WORDS)),
        "has_suspicious_extension": int(any(
            parsed.path.lower().endswith(e) or e in parsed.path.lower()
            for e in SUSPICIOUS_EXTENSIONS
        )),
        "has_shortener_domain": int(registered_domain in SHORTENERS),
        "has_hyphen_in_domain": int("-" in lower_domain),
        "has_brand_name": int(any(b in lower for b in BRAND_NAMES)),
        "is_free_hosting": int(registered_domain in FREE_HOSTING),
        "has_multiple_subdomains": int(len(subdomain_parts) >= 2),
    }
    return {name: float(values[name]) for name in FEATURE_NAMES}


def extract_feature_matrix(urls):
    import pandas as pd
    rows = [extract_url_features(url) for url in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)