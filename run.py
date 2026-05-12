"""Entry point for the Flask development server.

This is the single command markers run to launch the web app for the
Milestone 2 demo. It wraps the Flask application factory in
`app/__init__.py:create_app` and starts the Werkzeug dev server on
the default host/port (127.0.0.1:5000).

Usage:
  python run.py

For an auto-reloading dev server with the in-browser debugger, use
the Flask CLI directly (skips this entry point):
  flask --app app run --debug

Routes mounted by `create_app()` cover all four Milestone 2 tasks:
  Task 1 (search)         -> GET  /search?q=...
  Task 2 (classifier)     -> GET/POST /product/<id>/review
  Task 3 (recommendations)-> GET  /product/<id>     (Similar items section)
  Task 4 (aspects)        -> GET  /product/<id>     (Customers mention pills)

See README.md for the full setup procedure and `docs/milestone2_report.md`
for the per-task design rationale.
"""

from app import create_app

if __name__ == "__main__":
    create_app().run()
