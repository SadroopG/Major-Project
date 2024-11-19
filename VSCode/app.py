from flask import Flask, render_template, request
import sounddevice as sd
import numpy as np
import scipy.fftpack
import threading
import queue

app = Flask(__name__)

SAMPLE_FREQ = 44100
WINDOW_SIZE = 44100
WINDOW_STEP = 21050
CONCERT_PITCH = 440
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
listening = False
note_queue = queue.Queue()

def find_closest_note(pitch):
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    return closest_note, closest_pitch

def audio_callback(indata, frames, time, status):
    global windowSamples
    if status:
        print(status)
    if any(indata):
        windowSamples = np.concatenate((windowSamples, indata[:, 0]))  # append new samples
        windowSamples = windowSamples[len(indata[:, 0]):]  # remove old samples
        magnitudeSpec = abs(scipy.fftpack.fft(windowSamples)[:len(windowSamples) // 2])

        for i in range(int(62 / (SAMPLE_FREQ / WINDOW_SIZE))):
            magnitudeSpec[i] = 0

        maxInd = np.argmax(magnitudeSpec)
        maxFreq = maxInd * (SAMPLE_FREQ / WINDOW_SIZE)
        closestNote, closestPitch = find_closest_note(maxFreq)

        note_queue.put(f"{closestNote} {maxFreq:.1f}/{closestPitch:.1f}")
    else:
        note_queue.put("no input")

def audio_thread():
    global listening
    global windowSamples
    windowSamples = [0 for _ in range(WINDOW_SIZE)]
    try:
        with sd.InputStream(channels=1, callback=audio_callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
            while listening:
                pass
    except Exception as e:
        print(str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/listen', methods=['POST'])
def start_listening():
    global listening
    if not listening:
        listening = True
        threading.Thread(target=audio_thread, daemon=True).start()
    return "Listening started"

@app.route('/stop', methods=['POST'])
def stop_listening():
    global listening
    listening = False
    return "Listening stopped"

@app.route('/note', methods=['GET'])
def get_note():
    try:
        note = note_queue.get_nowait()
    except queue.Empty:
        note = "Waiting for input..."
    return note

if __name__ == '__main__':
    app.run(debug=True)
