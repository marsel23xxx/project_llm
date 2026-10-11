from redis.asyncio import Redis


async def hit(redis: Redis, key: str, limit: int, window_seconds: int) -> bool:
    """Count one attempt for `key`; return False once `limit` is exceeded within the window."""
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_seconds)
    return count <= limit


async def reset(redis: Redis, key: str) -> None:
    await redis.delete(key)
