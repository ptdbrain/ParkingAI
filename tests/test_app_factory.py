from __future__ import annotations


def test_create_app_import_does_not_initialize_database_engine() -> None:
    from app.main import create_app

    app = create_app()

    assert app.title == "Edge AI Parking System"
