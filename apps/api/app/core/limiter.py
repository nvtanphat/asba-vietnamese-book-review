"""Shared rate-limiter instance.

Kept in its own module (rather than defined in main.py) so routers can import it to
decorate individual endpoints without a circular import back to the app factory.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
