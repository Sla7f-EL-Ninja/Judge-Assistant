#!/usr/bin/env python3
"""
export_openapi.py

Generate OpenAPI spec files (JSON + YAML) from the FastAPI app without
starting a server or connecting to a database.

Usage:
    python -m api.scripts.export_openapi          # from project root
    python api/scripts/export_openapi.py           # also works

Output:
    docs/openapi.json
    docs/openapi.yaml
"""

import json
import os
import sys

# Ensure project root is on sys.path so `api.*` imports resolve
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    # Import the app factory -- this does NOT trigger the lifespan (no DB needed)
    from api.app import create_app

    app = create_app()
    spec = app.openapi()

    # Write output
    docs_dir = os.path.join(_project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    json_path = os.path.join(docs_dir, "openapi.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")

    # YAML output (optional -- skip if pyyaml not installed)
    try:
        import yaml

        yaml_path = os.path.join(docs_dir, "openapi.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Wrote {yaml_path}")
    except ImportError:
        print("pyyaml not installed -- skipping YAML export")


if __name__ == "__main__":
    main()
