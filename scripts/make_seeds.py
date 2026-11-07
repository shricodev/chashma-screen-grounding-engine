import argparse
import io
import re
import zipfile
from pathlib import Path

import httpx

TRANCO = "https://tranco-list.eu/top-1m.csv.zip"

BLOCK = re.compile(
    r"porn|xxx|sex|xvideo|xnxx|xhamster|hentai|nsfw|onlyfans|chaturbate|stripchat|livejasmin"
    r"|bongacams|spankbang|redtube|youporn|tube8|eporner|rule34|escort|camsoda|fapello|erotic"
    r"|fetlife|dlsite|furaffinity|f95zone|literotica|motherless|beeg|tnaflix|brazzers|bangbros"
    r"|clips4sale|manyvids|fansly|adult|nude"
    r"|cdn|static|api\.|apis\.|analytics|tracking|metrics|telemetry|doubleclick|adservice"
    r"|adsystem|adnxs|akamai|cloudfront|edgekey|edgesuite|fastly|gstatic|googleusercontent"
    r"|amazonaws|azureedge|windowsupdate|digicert|ocsp|ntp\.|dns\.|in-addr|gvt\d"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("data/seeds_tranco.txt"))
    args = ap.parse_args()

    resp = httpx.get(TRANCO, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        rows = zf.read(zf.namelist()[0]).decode().splitlines()

    seen_names: set[str] = set()
    domains = []
    for row in rows:
        domain = row.split(",", 1)[1].strip().lower()
        name = domain.split(".")[0]
        # google.com / google.de / google.co.uk all look the same
        if BLOCK.search(domain) or name in seen_names:
            continue
        seen_names.add(name)
        domains.append(domain)
        if len(domains) >= args.top:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(domains) + "\n")
    print(f"{len(domains)} seeds -> {args.out}")


if __name__ == "__main__":
    main()
