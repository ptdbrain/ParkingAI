# Migrations

Alembic migrations live here for production deployments.

The demo still calls `Base.metadata.create_all()` so local mock runs remain immediately runnable. For a managed database, set `PARKING_DATABASE_URL` and run:

```bash
alembic upgrade head
```
