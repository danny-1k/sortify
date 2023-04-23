from src import app


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", default=False, type=bool)

    args = parser.parse_args()

    PORT = os.environ.get("SERVER_PORT") or 3000

    app.run(port=PORT, debug=args.debug)
