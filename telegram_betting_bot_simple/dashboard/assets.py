"""Content-hashed static files: the page names every file by a hash of its bytes, so files can be cached for good.

At startup every file under dashboard/static is hashed. index.html is served with:

- an import map (``<script type="importmap">``) that maps each module's URL to ``URL?v=<hash>``, so the
  modules' own relative imports (``import ... from "./api.js"``) load the hashed URL too;
- ``<link rel="modulepreload">`` for the modules Today needs first, so the browser fetches them side by side
  instead of discovering them one import at a time;
- hashed addresses for the files the page links itself (styles.css, app.js).

A request whose ``?v=`` matches the file's current hash may be cached for a year (``immutable``); anything
else, and index.html, is ``no-cache`` (the browser checks back, and gets a 304 when nothing changed). A file
changed on disk without a restart is hashed again when next asked for, so an old hash never names new bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PREFIX = "/static/"
HASH_LENGTH = 12
IMMUTABLE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"
# The modules Today's first paint needs (preloaded); the optional ones only when they exist
CORE_MODULES = ("app.js", "js/app.js", "js/api.js", "js/store.js", "js/ui.js", "js/icons.js", "js/telegram.js",
                "js/views/today.js")
# (Venue & conditions' badges are drawn on Today's rows and the slip's legs: preloaded with them)
OPTIONAL_CORE = ("js/cache.js", "js/layout.js", "js/views/context/badges.js", "js/views/context/units.js",
                 "js/views/context/styles.js")
CORE_FOLDERS = ("js/views/today",)  # Today's own parts
LINK = re.compile(r'(?P<attr>\b(?:src|href))="/static/(?P<path>[^"?#]+)(?:\?[^"#]*)?"')
FIRST_TAG = re.compile(r"<(?:link|script)\b", re.IGNORECASE)


class StaticAssets:
    """The static folder's files by hash, and the page that names them."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._lock = threading.Lock()
        self._files: Dict[str, Tuple[int, int, str]] = {}  # relative path -> (mtime_ns, size, hash)
        self._index: Optional[Tuple[Tuple[Tuple[str, str], ...], bytes, str]] = None
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                self.version(path.relative_to(self.root).as_posix())

    def _path(self, rel: str) -> Optional[Path]:
        path = (self.root / rel).resolve()
        if self.root not in path.parents or not path.is_file():
            return None
        return path

    def version(self, rel: str) -> Optional[str]:
        """The file's content hash (hashed again when it changed on disk), or None when there is no such file."""
        path = self._path(rel)
        if path is None:
            with self._lock:
                self._files.pop(rel, None)
            return None
        stat = path.stat()
        with self._lock:
            known = self._files.get(rel)
            if known and known[:2] == (stat.st_mtime_ns, stat.st_size):
                return known[2]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:HASH_LENGTH]
        with self._lock:
            self._files[rel] = (stat.st_mtime_ns, stat.st_size, digest)
        return digest

    def url(self, rel: str) -> str:
        """"/static/<rel>?v=<hash>" (no ?v= for a file that isn't there)."""
        digest = self.version(rel)
        return f"{PREFIX}{rel}?v={digest}" if digest else f"{PREFIX}{rel}"

    def cache_control(self, rel: str, query_version: Optional[str]) -> str:
        """Immutable for the hash the file has now; no-cache for anything else."""
        if query_version and query_version == self.version(rel):
            return IMMUTABLE
        return NO_CACHE

    def modules(self) -> List[str]:
        """Every JavaScript module under the folder, by relative path."""
        with self._lock:
            known = sorted(self._files)
        # Files added since startup are picked up too
        found = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*.js") if path.is_file()}
        return sorted(found | {rel for rel in known if rel.endswith(".js") and self._path(rel)})

    def preloads(self) -> List[str]:
        core: List[str] = [rel for rel in CORE_MODULES + OPTIONAL_CORE if self._path(rel)]
        for folder in CORE_FOLDERS:
            base = self.root / folder
            if base.is_dir():
                core += sorted(p.relative_to(self.root).as_posix() for p in base.glob("*.js") if p.is_file())
        return list(dict.fromkeys(core))

    def import_map(self) -> Dict[str, Dict[str, str]]:
        return {"imports": {f"{PREFIX}{rel}": self.url(rel) for rel in self.modules()}}

    def _versions(self, rels: Iterable[str]) -> Tuple[Tuple[str, str], ...]:
        return tuple((rel, self.version(rel) or "") for rel in sorted(set(rels)))

    def index(self) -> Tuple[bytes, str]:
        """index.html with the import map, the module preloads and hashed links, and its ETag (quoted).
        Rebuilt only when a file it names changed."""
        modules = self.modules()
        linked = LINK.findall(self._read("index.html"))
        signature = self._versions(["index.html", *modules, *(path for _, path in linked)])
        cached = self._index
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        html = self._read("index.html")
        html = LINK.sub(lambda m: f'{m.group("attr")}="{self.url(m.group("path"))}"'
                        if self._path(m.group("path")) else m.group(0), html)
        head = "\n  ".join([
            '<script type="importmap">' + json.dumps(self.import_map(), separators=(",", ":")).replace("</", "<\\/")
            + "</script>",
            *(f'<link rel="modulepreload" href="{self.url(rel)}">' for rel in self.preloads()),
        ]) + "\n  "
        first = FIRST_TAG.search(html)
        at = first.start() if first else html.lower().find("</head>")
        if at < 0:
            at = 0
        html = html[:at] + head + html[at:]
        body = html.encode("utf-8")
        etag = '"' + hashlib.sha256(body).hexdigest()[:24] + '"'
        self._index = (signature, body, etag)
        return body, etag

    def _read(self, rel: str) -> str:
        path = self._path(rel)
        return path.read_text(encoding="utf-8") if path else ""
