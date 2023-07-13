import queue
import time
import urllib.request
from pathlib import Path
import sys
import numpy as np
import pydub
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import streamlit as st
import pywhisper
import pyaudio
import wave
import translate
from translate import Translator
from gtts import gTTS
import gtts
import deepspeech
from tempfile import NamedTemporaryFile
import os
from deepspeech import Model


path_parent = Path(__file__).parent


print('pywhisper',pywhisper.__version__)
print('streamlit',st.__version__)
print('gtts',gtts.__version__)
print('pyaudio',pyaudio.__version__)
print('numpy',np.__version__)
#print('pydub',pydub.__version__)
#print('deepspeech',deepspeech.__version__)
print('translate',translate.__version__)
#print('wave',wave.__version__)



audio = None



def main():
    model_link = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    language_model_link = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    model_path = path_parent / "models/deepspeech-0.9.3-models.pbmm"
    language_model_path = path_parent / "models/deepspeech-0.9.3-models.scorer"

    # download_model(model_link, model_path, expected_size=188915987)
    # download_model(language_model_link, language_model_path, expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    model_path = str(model_path)
    language_model_path = str(language_model_path)

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )
    status_indicator = st.empty()

    print(webrtc_ctx.state.playing)
    if not webrtc_ctx.state.playing:
        return
    print('started')

    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.audio_receiver:
            print('webrtc start')
            if stream is None:

                model = Model(model_path)
                model.enableExternalScorer(language_model_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

                print('model completed')

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Start Talking.....")
            print('running')
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
                st.session_state.text_inp = text
        else:
            print('abort after stopping')
            status_indicator.write("AudioReciver is not set. Abort.")
            break


def download_model(url, download_to, expected_size=None):
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def text_to_speech(text, lang='en'):
    tts = gTTS(text, lang=lang)
    with NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        file_name = f.name
        tts.save(file_name)
    return file_name

st.title("Voice to Text Conversion")
audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])

def loadmodel():
    model = pywhisper.load_model("base")
    return model

if st.sidebar.button("Load Pywhisper Model"):
    model = loadmodel()
    st.sidebar.success("PyWhisper Model Load Completed Successfully")


if st.sidebar.button("Transcribe Audio"):
    if audio is not None:
        model = loadmodel()
        st.sidebar.success("Transcribing Audio")
        try:
            transcribe = model.transcribe(audio.name)
        except:
            st.sidebar.error("Please record audio first")

        st.sidebar.success("Transcription Completed")
        st.text(transcribe["text"])
        st.session_state.text_inp = transcribe["text"]
    else:
        st.sidebar.error('Please upload audio file or record audio')


if "text_inp" not in st.session_state:
    st.session_state.text_inp = ""


if 'text_inp' in st.session_state and st.session_state.text_inp:
    st.write(st.session_state.text_inp)
    language = st.selectbox("Select language for the output speech:", [
                ("English", "en"),
                ("Spanish", "es"),
                ("French", "fr"),
                ("German", "de"),
                ("Italian", "it"),
                ("Portuguese", "pt"),
                ("Russian", "ru")
            ], format_func=lambda x: x[0])

    st.write("Selected language:", language[0])

    translator = Translator(to_lang=language[1])
    translation = translator.translate(st.session_state.text_inp)
    print(st.session_state.text_inp)

    if st.button("Convert Text to Speech"):
        audio_file = text_to_speech(translation, lang=language[1])
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
            os.unlink(audio_file)
        else:
            st.write("Error converting text to speech")


if __name__ == "__main__":    
    main()