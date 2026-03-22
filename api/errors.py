"""
errors.py

Machine-readable error code constants used across the API.

Every error response includes one of these codes in the ``error.code`` field
so that clients can switch on a stable string instead of parsing human-readable
messages.
"""

# -- Authentication / authorisation -------------------------------------------
UNAUTHORIZED = "UNAUTHORIZED"

# -- Validation ---------------------------------------------------------------
VALIDATION_ERROR = "VALIDATION_ERROR"

# -- Resource not found -------------------------------------------------------
CASE_NOT_FOUND = "CASE_NOT_FOUND"
CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
FILE_NOT_FOUND = "FILE_NOT_FOUND"
SUMMARY_NOT_FOUND = "SUMMARY_NOT_FOUND"

# -- File upload --------------------------------------------------------------
INVALID_MIME_TYPE = "INVALID_MIME_TYPE"
FILE_TOO_LARGE = "FILE_TOO_LARGE"

# -- Business logic -----------------------------------------------------------
NO_FIELDS_TO_UPDATE = "NO_FIELDS_TO_UPDATE"

# -- Server -------------------------------------------------------------------
INTERNAL_ERROR = "INTERNAL_ERROR"
