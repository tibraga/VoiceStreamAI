import json
import logging
import ssl
import uuid

import websockets

from src.client import Client
from src.asr.asr_factory import ASRFactory
from src.vad.vad_factory import VADFactory


class Server:
    """
    Represents the WebSocket server for handling real-time audio transcription.

    This class manages WebSocket connections, processes incoming audio data,
    and interacts with VAD and ASR pipelines for voice activity detection and
    speech recognition.
    """

    def __init__(
        self,
        vad_type,
        vad_args,
        asr_type,
        asr_args,
        host="localhost",
        port=8765,
        sampling_rate=16000,
        samples_width=2,
        certfile=None,
        keyfile=None,
    ):
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.certfile = certfile
        self.keyfile = keyfile
        self.connected_clients = {}

        # Number of GPUs available
        self.num_gpus = 4  # You can set this dynamically if needed

        # Initialize pipelines for each GPU
        self.vad_pipelines = []
        self.asr_pipelines = []

        for i in range(self.num_gpus):
            # Set device index in args
            vad_args_copy = vad_args.copy()
            asr_args_copy = asr_args.copy()

            vad_args_copy['device'] = 'cuda'
            vad_args_copy['device_index'] = i
            vad_pipeline = VADFactory.create_vad_pipeline(vad_type, **vad_args_copy)
            self.vad_pipelines.append(vad_pipeline)

            asr_args_copy['device'] = 'cuda'
            asr_args_copy['device_index'] = i
            asr_pipeline = ASRFactory.create_asr_pipeline(asr_type, **asr_args_copy)
            self.asr_pipelines.append(asr_pipeline)

        self.pipeline_index = 0  # To keep track of pipeline assignment

    async def handle_audio(self, client, websocket, vad_pipeline, asr_pipeline):
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
                websocket, vad_pipeline, asr_pipeline
            )

    async def handle_websocket(self, websocket):
        client_id = str(uuid.uuid4())

        # Assign ASR and VAD pipelines to client
        pipeline_index = self.pipeline_index % self.num_gpus
        self.pipeline_index += 1

        assigned_vad_pipeline = self.vad_pipelines[pipeline_index]
        assigned_asr_pipeline = self.asr_pipelines[pipeline_index]

        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client

        print(f"Client {client_id} connected, assigned to GPU {pipeline_index}")

        try:
            await self.handle_audio(client, websocket, assigned_vad_pipeline, assigned_asr_pipeline)
        except websockets.ConnectionClosed as e:
            print(f"Connection with {client_id} closed: {e}")
        finally:
            del self.connected_clients[client_id]

    def start(self):
        if self.certfile:
            # Create an SSL context to enforce encrypted connections
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

            # Load your server's certificate and private key
            # Replace 'your_cert_path.pem' and 'your_key_path.pem' with the
            # actual paths to your files
            ssl_context.load_cert_chain(
                certfile=self.certfile, keyfile=self.keyfile
            )

            print(
                f"WebSocket server ready to accept secure connections on "
                f"{self.host}:{self.port}"
            )

            # Pass the SSL context to the serve function along with the host
            # and port. Ensure the secure flag is set to True if using a secure
            # WebSocket protocol (wss://)
            return websockets.serve(
                self.handle_websocket, self.host, self.port, ssl=ssl_context
            )
        else:
            print(
                f"WebSocket server ready to accept connections on "
                f"{self.host}:{self.port}"
            )
            return websockets.serve(
                self.handle_websocket, self.host, self.port
            )
