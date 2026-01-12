import argparse

from cli.app import run as run_cli
from http_server.app import run as run_http_server


def main():
    parser = argparse.ArgumentParser(description="CommitTutor entrypoint")
    parser.add_argument(
        "--http_server",
        action="store_true",
        help="Run CommitTutor as an HTTP server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP server host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port.")
    args = parser.parse_args()

    if args.http_server:
        run_http_server(args.host, args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
