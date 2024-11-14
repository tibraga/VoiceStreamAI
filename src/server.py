import json
import logging
import ssl
import uuid

import websockets

from src.client import Client


class Server:
    """
    Represents the WebSocket server for handling real-time audio transcription.

    This class manages WebSocket connections, processes incoming audio data,
    and interacts with VAD and ASR pipelines for voice activity detection and
    speech recognition.

    Attributes:
        vad_pipeline: An instance of a voice activity detection pipeline.
        asr_pipeline: An instance of an automatic speech recognition pipeline.
        host (str): Host address of the server.
        port (int): Port on which the server listens.
        sampling_rate (int): The sampling rate of audio data in Hz.
        samples_width (int): The width of each audio sample in bits.
        connected_clients (dict): A dictionary mapping client IDs to Client
                                  objects.
    """

    def __init__(
        self,
        vad_pipeline,
        asr_pipeline,
        host="localhost",
        port=8765,
        sampling_rate=16000,
        samples_width=2,
        certfile=None,
        keyfile=None,
    ):
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.certfile = certfile
        self.keyfile = keyfile
        self.connected_clients = {}

    async def handle_audio(self, client, websocket):
        while True:
            message = await websocket.recv()

            if isinstance(message, bytes):
                client.append_audio_data(message)
            elif isinstance(message, str):
                config = json.loads(message)
                if config.get("type") == "config":
                    client.update_config(config["data"])
                    logging.debug(f"Updated config: {client.config}")
                    continue
            else:
                print(f"Unexpected message type from {client.client_id}")

            # this is synchronous, any async operation is in BufferingStrategy
            client.process_audio(
                websocket, self.vad_pipeline, self.asr_pipeline
            )

    async def handle_websocket(self, websocket):
        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        print(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        except websockets.ConnectionClosed as e:
            print(f"Connection with {client_id} closed: {e}")
        finally:
            del self.connected_clients[client_id]

    async def process_request(self, path, request_headers):
        """
        Handle incoming HTTP requests before the WebSocket handshake.

        Args:
            path (str): The request path.
            request_headers (dict): The request headers.

        Returns:
            A tuple of (HTTP status code, response headers, response body)
            if handling an HTTP request, or None to continue with WebSocket.
        """
        if path == '/healthcheck':
            # Respond with a 200 OK for the health check
            return (
                200,
                [('Content-Type', 'text/plain')],
                b'OK'
            )
        else:
            # Return None to proceed with the WebSocket handshake
            return None

    def start(self):
        print("AQUI!")
        ssl_context = None
        if self.certfile:
            # Create an SSL context to enforce encrypted connections
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(
                certfile=self.certfile, keyfile=self.keyfile
            )

        print(
            f"WebSocket server ready to accept connections on "
            f"{self.host}:{self.port}"
        )

        return websockets.serve(
            self.handle_websocket,
            self.host,
            self.port,
            ssl=ssl_context,
            process_request=self.process_request  # Add this line
        )