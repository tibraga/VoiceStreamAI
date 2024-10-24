import pathlib
from gevent.pool import Pool
from locust import User, task
from locust.exception import StopUser
from pydub import AudioSegment
from websockets.sync.client import connect
from websockets.exceptions import InvalidMessage

import os
import time
import logging
import json
import gevent


logging.basicConfig(level=logging.INFO)


class WebSocketUser(User):
    abstract = True

    def __init__(self, environment):
        super().__init__(environment)
        self.pool = Pool(100)
        with self.environment.events.request.measure("[Connect]", "Websocket"):
            self.client = connect(self.host)
            self.client.send('{"type":"config","data":{"sampleRate":48000,"channels":1,"language":"portuguese","processing_strategy":"silence_at_end_of_chunk","processing_args":{"chunk_length_seconds":2,"chunk_offset_seconds":0.05}}}')
            gevent.sleep(0.25)


    def on_start(self):

        def _receive():
            while True:
                try:
                    transcription_str = self.client.recv()
                except InvalidMessage as e:
                    logging.error("Invalid message:", e)
                except Exception as e:
                    logging.error("Error:", e)
                    break
                else:
                    client_id = self.client.id
                    transcription_end = time.time()
                    time_elapse = round(transcription_end - self.start_time, 2)
                    transcription_json = json.loads(transcription_str)
                    logging.debug(transcription_json)
                    logging.info(
                        f"[{client_id}] Time elapse: {time_elapse}s, Received: {transcription_json['text']}"
                    )

                    self.environment.events.request.fire(
                        request_type="[Receive]",
                        name="Response",
                        response_time=time_elapse * 1000,  # Transformar para milissegundos
                        response_length=0,  # Caso queira adicionar o tamanho da resposta
                        exception=None  # Indica que a métrica é bem-sucedida
                    )

        self.pool.spawn(_receive)

    def on_stop(self):
        super().on_stop()
        logging.info("Closing websocket connection")
        self.pool.kill()
        self.client.close()

    @task
    def send_streaming_audio(self):
        for filename in os.listdir(self.audio_file_path):
            self.start_time = time.time()
            if filename.endswith(".wav"):
                audio_file = os.path.join(self.audio_file_path, filename)
                logging.info(f"Loading audio file: {audio_file}")

                with open(audio_file, "rb") as file:
                    with self.environment.events.request.measure(
                        "[Send]", "Audio sents"
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
                        for i in range(0, len(audio), 250):
                            chunk = audio[i : i + 250]
                            logging.debug(f"Sending trunk {i}...")
                            self.client.send(chunk.raw_data)
                            gevent.sleep(0.25)
        gevent.sleep(60)
        raise StopUser()


class EnglisthStreamWhisperWebSocketUser(WebSocketUser):
    host = "ws://ws.reasoner.alpha.sofya.ai:8043"
    audio_file_path = "./data/en"
    start_time: float = 0.0