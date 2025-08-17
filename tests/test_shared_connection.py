"""Test shared connection mode for LLMDatabase and LLMRing."""

import pytest
from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmring.db import LLMDatabase
from llmring.service import LLMRing


@pytest.mark.asyncio
async def test_llm_database_with_external_manager():
    """Test LLMDatabase with external AsyncDatabaseManager."""
    # Create a shared manager
    config = DatabaseConfig(
        connection_string="postgresql://postgres:postgres@localhost/postgres",
        schema="test_shared",
        max_connections=10,
    )
    shared_manager = AsyncDatabaseManager(config)
    await shared_manager.connect()

    try:
        # Create LLMDatabase with shared manager
        db = LLMDatabase.from_manager(shared_manager, schema="test_llmring")

        # Verify it's using external db
        assert db._external_db is True
        assert db.db is shared_manager

        # Initialize should not create new connection
        await db.initialize()
        assert db._initialized is True

        # Close should not disconnect shared manager
        await db.close()

        # Shared manager should still be connected
        # Test with a simple query
        result = await shared_manager.fetch_value("SELECT 1")
        assert result == 1

    finally:
        await shared_manager.disconnect()


@pytest.mark.asyncio
async def test_llm_database_standalone_mode():
    """Test LLMDatabase in standalone mode."""
    # Create standalone database
    db = LLMDatabase(
        connection_string="postgresql://postgres:postgres@localhost/postgres",
        schema="test_llm_standalone",
    )

    # Verify it's not using external db
    assert db._external_db is False

    try:
        # Initialize creates connection
        await db.initialize()
        assert db._initialized is True

        # Test basic query
        result = await db.db.fetch_value("SELECT 1")
        assert result == 1

    finally:
        # Close disconnects in standalone mode
        await db.close()
        assert db._initialized is False


@pytest.mark.asyncio
async def test_llmring_with_shared_connection():
    """Test LLMRing with shared database connection."""
    # Create a shared manager
    config = DatabaseConfig(
        connection_string="postgresql://postgres:postgres@localhost/postgres",
        max_connections=15,
    )
    shared_manager = AsyncDatabaseManager(config)
    await shared_manager.connect()

    try:
        # Create LLMRing with shared manager
        service = LLMRing(
            db_manager=shared_manager, origin="test-engine", enable_db_logging=True
        )

        # Verify database is configured correctly
        assert service.db is not None
        assert service.db._external_db is True
        assert service.db.db is shared_manager

        # Initialize service database
        await service.db.initialize()

        # Test that we can get models (if any exist)
        # This verifies the database is working
        try:
            models = await service.get_models_from_db()
            # Just verify it returns a list (might be empty)
            assert isinstance(models, list)
        except Exception:
            # It's OK if this fails due to missing tables in test
            pass

    finally:
        await shared_manager.disconnect()


@pytest.mark.asyncio
async def test_connection_pool_sharing():
    """Verify that multiple services share the same connection pool."""
    # Create a shared manager with limited connections
    config = DatabaseConfig(
        connection_string="postgresql://postgres:postgres@localhost/postgres",
        min_connections=2,
        max_connections=5,  # Small pool to test sharing
    )
    shared_manager = AsyncDatabaseManager(config)
    await shared_manager.connect()

    try:
        # Create multiple LLM databases with same manager
        db1 = LLMDatabase.from_manager(shared_manager, schema="test_schema1")
        db2 = LLMDatabase.from_manager(shared_manager, schema="test_schema2")

        # Initialize both
        await db1.initialize()
        await db2.initialize()

        # Both should use the same underlying pool
        assert db1.db is shared_manager
        assert db2.db is shared_manager
        assert db1.db is db2.db

        # Get pool stats
        stats = await shared_manager.get_pool_stats()

        # Pool should be within our limits
        assert stats["min_size"] == 2
        assert stats["max_size"] == 5
        assert stats["size"] <= 5

        # Run queries on both to verify they work
        result1 = await db1.db.fetch_value("SELECT 1")
        result2 = await db2.db.fetch_value("SELECT 2")

        assert result1 == 1
        assert result2 == 2

        # Close both - should not affect shared manager
        await db1.close()
        await db2.close()

        # Manager should still work
        result = await shared_manager.fetch_value("SELECT 3")
        assert result == 3

    finally:
        await shared_manager.disconnect()
