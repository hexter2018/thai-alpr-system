"""
Database Configuration and Session Management
"""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import Generator, AsyncGenerator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
            echo: Echo SQL statements
        """
        self.database_url = database_url
        self.echo = echo
        
        # Convert postgres:// to postgresql:// for asyncpg
        self.async_database_url = database_url.replace(
            "postgresql://",
            "postgresql+asyncpg://"
        )
        
        # Sync engine
        self.engine = create_engine(
            database_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Async engine
        self.async_engine = create_async_engine(
            self.async_database_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Session makers
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self.AsyncSessionLocal = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Database manager initialized")
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get synchronous database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get asynchronous database session"""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
    
    @asynccontextmanager
    async def session_scope(self):
        """Provide a transactional scope for async operations"""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def test_async_connection(self) -> bool:
        """Test async database connection"""
        try:
            async with self.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Async database connection successful")
            return True
        except Exception as e:
            logger.error(f"Async database connection failed: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        self.engine.dispose()
        logger.info("Database connections closed")
    
    async def async_close(self):
        """Close async database connections"""
        await self.async_engine.dispose()
        logger.info("Async database connections closed")


# Global database manager
_db_manager: DatabaseManager = None


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """
    Initialize database manager
    
    Args:
        database_url: Database connection URL
        echo: Echo SQL statements
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo)
    return _db_manager


def get_db_manager() -> DatabaseManager:
    """Get database manager instance"""
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_manager


# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    db_manager = get_db_manager()
    yield from db_manager.get_session()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    db_manager = get_db_manager()
    async for session in db_manager.get_async_session():
        yield session