import asyncio
import json
import os
import time

from .buffering_strategy_interface import BufferingStrategyInterface


class UniqueChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio at the end of each chunk with
    silence detection.

    This class is responsible for handling audio chunks, detecting silence at
    the end of each chunk, and initiating the transcription process for the
    chunk.

    Attributes:
        client (Client): The client instance associated with this buffering
                         strategy.
        chunk_length_seconds (float): Length of each audio chunk in seconds.
        chunk_offset_seconds (float): Offset time in seconds to be considered
                                      for processing audio chunks.
    """

    def __init__(self, client, **kwargs):
        """
        Initialize the UniqueChunk buffering strategy.

        Args:
            client (Client): The client instance associated with this buffering
                             strategy.
            **kwargs: Additional keyword arguments
        """
        self.client = client
        self.processing_flag = False

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        """
        Process unique audio chunks and scheduling
        asynchronous processing.


        Args:
            websocket: The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        self.client.scratch_buffer += self.client.buffer
        self.client.buffer.clear()

        self.processing_flag = True
        # Schedule the processing in a separate task
        asyncio.create_task(
            self.process_audio_async(websocket, asr_pipeline)
        )

    async def process_audio_async(self, websocket, asr_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity
        detection and transcription of the audio data. It sends the
        transcription results through the WebSocket connection.

        Args:
            websocket (Websocket): The WebSocket connection for sending
                                   transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        start = time.time()
    
        transcription = await asr_pipeline.transcribe(self.client)
        if transcription["text"] != "":
            end = time.time()
            transcription["processing_time"] = end - start
            json_transcription = json.dumps(transcription)
            await websocket.send(json_transcription)
            self.client.scratch_buffer.clear()
            self.client.increment_file_counter()

        self.processing_flag = False
