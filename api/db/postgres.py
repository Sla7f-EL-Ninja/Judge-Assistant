"""
postgres.py

PostgreSQL database management via SQLAlchemy (async).

Provides user management, role-based access control (RBAC), and audit
logging for legal compliance.

``connect_postgres`` / ``close_postgres`` are called from the FastAPI lifespan.
``get_async_session`` is a FastAPI dependency that yields a session.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    relationship,
    sessionmaker,
)

from config.api import Settings

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[sessionmaker] = None


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all PostgreSQL models."""
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class User(Base):
    """User accounts for the Judge Assistant system."""

    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=lambda: f"user_{uuid.uuid4().hex[:12]}")
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(
        Enum("judge", "clerk", "admin", "viewer", name="user_role"),
        nullable=False,
        default="viewer",
    )
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    audit_logs = relationship("AuditLog", back_populates="user", lazy="dynamic")
    permissions = relationship("UserPermission", back_populates="user", lazy="joined")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"


class UserPermission(Base):
    """Fine-grained permissions for users beyond their role defaults."""

    __tablename__ = "user_permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    permission = Column(
        Enum(
            "cases:read", "cases:write", "cases:delete",
            "files:read", "files:write", "files:delete",
            "queries:execute", "conversations:read", "conversations:delete",
            "summaries:read", "users:manage", "audit:read",
            name="permission_type",
        ),
        nullable=False,
    )
    granted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    granted_by = Column(String(64), nullable=True)

    # Relationships
    user = relationship("User", back_populates="permissions")

    def __repr__(self) -> str:
        return f"<UserPermission user={self.user_id} perm={self.permission}>"


class AuditLog(Base):
    """Audit trail for all user actions -- required for legal compliance.

    Records who did what, when, and on which resource.
    """

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False, index=True)
    resource_id = Column(String(255), nullable=True)
    detail = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self) -> str:
        return f"<AuditLog user={self.user_id} action={self.action} resource={self.resource_type}/{self.resource_id}>"


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

async def connect_postgres(settings: Settings) -> None:
    """Create the async engine and session factory, and create tables."""
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.postgresql_url,
        pool_size=settings.postgresql_pool_size,
        max_overflow=settings.postgresql_max_overflow,
        echo=settings.debug,
    )

    _session_factory = sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create all tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("PostgreSQL connected and tables created")


async def close_postgres() -> None:
    """Dispose of the engine and close all connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session. Use as a FastAPI dependency."""
    if _session_factory is None:
        raise RuntimeError("PostgreSQL is not connected. Call connect_postgres first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Audit logging helper
# ---------------------------------------------------------------------------

async def log_audit_event(
    session: AsyncSession,
    user_id: Optional[str],
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    detail: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """Record an audit log entry.

    Parameters
    ----------
    session : AsyncSession
        The active database session.
    user_id : str or None
        The user who performed the action.
    action : str
        What was done (e.g. 'create_case', 'delete_conversation', 'query').
    resource_type : str
        The type of resource affected (e.g. 'case', 'file', 'conversation').
    resource_id : str or None
        The specific resource identifier.
    detail : str or None
        Additional context about the action.
    ip_address : str or None
        Client IP address.
    user_agent : str or None
        Client user agent string.
    """
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        detail=detail,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.add(entry)
    await session.flush()
