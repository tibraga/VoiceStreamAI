import pathlib
from locust import User, task, between
from locust.exception import StopUser
from pydub import AudioSegment
from websockets.sync.client import connect
from websockets.exceptions import InvalidMessage

import os
import time
import logging
import json
import gevent
import random

logging.basicConfig(level=logging.INFO)


class WebSocketUser(User):
    abstract = True
    wait_time = between(1, 3)  # Interval between tasks to vary the traffic

    def on_start(self):
        # Any initialization needed when a user starts
        pass

    def on_stop(self):
        logging.info("Stopping user")
        # Any cleanup needed when a user stops

    @task
    def send_streaming_audio(self):
        # Select a WebSocket server for this user
        current_host = self.select_host()

        logging.info(f"User {id(self)} connecting to WebSocket server: {current_host}")

        # Create a new WebSocket connection for each task execution
        try:
            with connect(current_host) as client:
                client.send('{"type":"config","data":{"sampleRate":48000,"channels":1,"language":"portuguese","processing_strategy":"UniqueChunk","processing_args":{"chunk_length_seconds":2,"chunk_offset_seconds":0.05}}}')
                gevent.sleep(0.25)

                for filename in os.listdir(self.audio_file_path):
                    self.start_time = time.time()
                    if filename.endswith(".wav") or filename.endswith(".mp3"):
                        audio_file = os.path.join(self.audio_file_path, filename)
                        logging.info(f"Loading audio file: {audio_file}")

                        with open(audio_file, "rb") as file:
                            with self.environment.events.request.measure(
                                "[Send]", "Audio sent"
                            ):
                                file_format = pathlib.Path(audio_file).suffix[1:]
                                logging.debug(f"File format: {file_format}")
                                try:
                                    audio = AudioSegment.from_file(file, format=file_format)
                                    audio = audio.set_sample_width(2)
                                    audio = audio.set_frame_rate(16000)
                                except Exception as e:
                                    logging.error("File loading error:", e)

                                logging.info("Start sending audio")
                                client.send(audio.raw_data)
                                gevent.sleep(1)

                # Receive the response and measure the time
                try:
                    transcription_str = client.recv()
                    transcription_json = json.loads(transcription_str)
                    transcription_end = time.time()
                    time_elapse = round(transcription_end - self.start_time, 2)
                    logging.info(
                        f"Time elapsed: {time_elapse}s, Received: {transcription_json['text']}"
                    )

                    self.environment.events.request.fire(
                        request_type="[Receive]",
                        name="Response",
                        response_time=time_elapse * 1000,  # In milliseconds
                        response_length=0,
                        exception=None
                    )
                except InvalidMessage as e:
                    logging.error("Invalid message:", e)
                except Exception as e:
                    logging.error("Error:", e)

        except Exception as e:
            logging.error("Connection error:", e)

        # Pause between executions
        gevent.sleep(60)
        raise StopUser()

    def select_host(self):
        # This method should be overridden in the subclass
        raise NotImplementedError("Please implement select_host() in the subclass")


class EnglisthStreamWhisperWebSocketUser(WebSocketUser):
    def __init__(self, parent):
        super().__init__(parent)
        # Initialize the list of WebSocket servers as an instance variable
        self.hosts_list = [
            "wss://7b9hvrxz5xrpbg-8000.proxy.runpod.net",
            "wss://dlzux8j9bi1fxo-8000.proxy.runpod.net"
        ]
        self.audio_file_path = "./data/en"
        self.start_time: float = 0.0

        # Seed the random number generator uniquely per user
        random.seed(os.getpid() + id(self) + time.time())

    def select_host(self):
        # Select a host for this user
        return random.choice(self.hosts_list)