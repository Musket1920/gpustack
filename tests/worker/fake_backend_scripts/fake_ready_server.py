import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--health-path", default="/v1/models")
    return parser.parse_args()


def main():
    args = parse_args()
    health_path = args.health_path

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == health_path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data": []}')
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
