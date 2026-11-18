"""Request-safety helpers for the visualization server.

The server has no authentication, so everything it serves is confined to the
search root, restricted to known file types, and only answered for requests
whose ``Host`` header names this machine.
"""

import ipaddress
import os
import socket
from typing import FrozenSet, Iterable, Optional

DEFAULT_HOST = "127.0.0.1"

# Files `/plot_file` may serve, and the content type each is sent with.
IMAGE_CONTENT_TYPES = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# The only files served from the webui package directory.
STATIC_FILES = frozenset(
    {
        "/index.html",
        "/viz_tree.html",
        "/compare.html",
        "/favicon.png",
        "/sakana.jpg",
    }
)

LOOPBACK_HOSTNAMES = frozenset({"localhost"})


def resolve_under_root(root: str, candidate: Optional[str]) -> Optional[str]:
    """Resolve ``candidate`` against ``root`` and confine it to ``root``.

    Relative candidates are joined to the root; absolute ones are taken as is.
    Symlinks are resolved on both sides, so a link inside the root that points
    outside it is rejected. Returns the real path, or ``None`` when the
    candidate is empty, malformed, or escapes the root.
    """
    if not candidate or "\x00" in candidate:
        return None
    real_root = os.path.realpath(root)
    real_path = _fully_resolved(os.path.join(real_root, candidate))
    if real_path is None:
        return None
    try:
        inside = os.path.commonpath([real_root, real_path]) == real_root
    except ValueError:  # e.g. different drives on Windows
        return None
    return real_path if inside else None


def _fully_resolved(path: str) -> Optional[str]:
    """``path`` with every symlink resolved, or ``None`` if that cannot be done.

    Before Python 3.13 a lenient ``realpath`` gives up at a symlink loop and
    returns the components after it unresolved, so ``loop/../link`` comes back
    as a path that still goes through ``link``. Strict resolution raises on a
    loop instead. Only a path that does not exist yet is resolved leniently,
    and then only accepted if resolving the result again changes nothing.
    """
    try:
        return os.path.realpath(path, strict=True)
    except (FileNotFoundError, NotADirectoryError):
        lenient = os.path.realpath(path)
        return lenient if os.path.realpath(lenient) == lenient else None
    except OSError:  # a symlink loop, or a component that cannot be read
        return None


def hostname_from_host_header(value: Optional[str]) -> Optional[str]:
    """Extract the lower-cased hostname from a ``Host`` header value.

    Accepts ``name``, ``name:port``, ``[v6]`` and ``[v6]:port``. Returns
    ``None`` for anything else.
    """
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return None
        host, rest = value[1:end], value[end + 1 :]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return None
    else:
        host, sep, port = value.partition(":")
        if sep and port and not port.isdigit():
            return None
    host = host.rstrip(".")
    return host or None


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def build_allowed_hosts(
    bind_host: str, extra_hosts: Iterable[str] = ()
) -> FrozenSet[str]:
    """Hostnames a request may address the server by.

    Loopback names, this machine's hostname, the bind address when it is a
    name, and any explicitly allowed extras. IP literals are always accepted
    by :func:`is_allowed_host` and need not be listed.
    """
    names = set(LOOPBACK_HOSTNAMES)
    try:
        names.add(socket.gethostname())
    except OSError:
        pass
    names.add(bind_host)
    names.update(extra_hosts)
    cleaned = (hostname_from_host_header(name) for name in names if name)
    return frozenset(name for name in cleaned if name)


def is_allowed_host(host_header: Optional[str], allowed_hosts: FrozenSet[str]) -> bool:
    """Whether a request's ``Host`` header names this server.

    Rejecting foreign names defeats DNS rebinding: a page on ``evil.example``
    that re-resolves its own name to this machine still sends
    ``Host: evil.example``. An IP literal cannot be rebound, so those pass. A
    request with no ``Host`` header is not from a browser and passes too.
    """
    if host_header is None:
        return True
    host = hostname_from_host_header(host_header)
    if host is None:
        return False
    return host in allowed_hosts or _is_ip_literal(host)
