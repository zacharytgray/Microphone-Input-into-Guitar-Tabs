import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import find_peaks
import time
from scipy.fft import fft

SAMPLE_RATE = 44100  # Hz (CD Quality)
FILENAME = "recording.wav"

def record_audio(duration):
    countdown(numSeconds=3, duration=duration)
    print("Recording!")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(FILENAME, SAMPLE_RATE, recording)
    print("Done Recording.")
    return FILENAME

def countdown(numSeconds, duration):
    print(f"\nGet ready to record for {duration} seconds in...")
    for i in range(numSeconds, 0, -1):
        print(f"{i}...")
        time.sleep(1)

def extract_pitch(audio_file, frame_size=2048, hop_length=512):
    # Load the audio file
    sr, audio = wavfile.read(audio_file)
    
    # Convert to mono if stereo
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)
    
    # Normalize audio
    audio = audio.astype(float)
    audio /= np.max(np.abs(audio))

    # Calculate the number of frames
    num_frames = 1 + (len(audio) - frame_size) // hop_length
    
    pitches = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_size
        frame = audio[start:end]
        
        # Apply Hann window
        frame = frame * np.hanning(frame_size)
        
        # Compute FFT
        fft_frame = fft(frame)
        magnitude = np.abs(fft_frame)[:frame_size//2]
        
        # Find peak frequency
        peak_index = np.argmax(magnitude)
        frequency = peak_index * sr / frame_size
        
        # Amplitude threshold to detect silence
        if np.max(np.abs(frame)) < 0.1:
            pitches.append(0)  # Silence
        else:
            pitches.append(frequency)
    
    return pitches, sr

def pitch_to_note(pitch):
    # Define note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    if pitch == 0:
        return "Rest"
    
    # Convert frequency to MIDI note number
    midi_note = 12 * np.log2(pitch / 440) + 69
    
    # Round to nearest integer
    midi_note = round(midi_note)
    
    # Get note name and octave
    note_name = note_names[midi_note % 12]
    octave = midi_note // 12 - 1
    
    return f"{note_name}{octave}"
    # return note_name

def smooth_notes(notes, window_size=5):
    smoothed = []
    for i in range(len(notes)):
        window = notes[max(0, i-window_size//2):i+window_size//2+1]
        if window.count(window[0]) > len(window) // 2:
            smoothed.append(window[0])
        else:
            smoothed.append(notes[i])
    return smoothed

def generate_notes(pitch_sequence, sr, hop_length=512):
    notes = [pitch_to_note(pitch) for pitch in pitch_sequence]
    
    # Smooth the notes
    smoothed_notes = smooth_notes(notes)
    
    # Combine consecutive identical notes
    combined_notes = []
    current_note = smoothed_notes[0]
    duration = 1
    for note in smoothed_notes[1:]:
        if note == current_note:
            duration += 1
        else:
            combined_notes.append((current_note, duration * hop_length / sr))
            current_note = note
            duration = 1
    combined_notes.append((current_note, duration * hop_length / sr))
    
    return combined_notes


def note_to_midi(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1])
    note_name = note[:-1]
    return notes.index(note_name) + (octave + 1) * 12

def midi_to_guitar_tab(midi_note):
    strings = ['E', 'A', 'D', 'G', 'B', 'e']
    open_strings = [40, 45, 50, 55, 59, 64]  # MIDI numbers for open strings
    
    best_string = 0
    best_fret = 100  # Start with an impossibly high fret
    
    for i, open_string in enumerate(open_strings):
        if midi_note >= open_string:
            fret = midi_note - open_string
            if fret <= 20 and fret < best_fret:  # Prefer lower frets
                best_string = i
                best_fret = fret
    
    if best_fret == 100:
        return "-", "-"
    else:
        return strings[best_string], str(best_fret)

def generate_guitar_tabs(notes):
    tabs = {string: [] for string in ['e', 'B', 'G', 'D', 'A', 'E']}
    
    for note, duration in notes:
        if note == "Rest":
            fret = "-"
            string = "-"
        else:
            midi_note = note_to_midi(note)
            string, fret = midi_to_guitar_tab(midi_note)
        
        # Represent duration: 1 character ≈ 0.1 seconds
        repeat = max(1, int(duration * 10))
        for s in tabs:
            if s == string:
                tabs[s].append(fret.rjust(2))
                tabs[s].extend(["-"] * (repeat - 1))
            else:
                tabs[s].extend(["-"] * repeat)
    
    # Ensure all strings have the same length
    max_length = max(len(frets) for frets in tabs.values())
    for string in tabs:
        tabs[string] = tabs[string] + ["-"] * (max_length - len(tabs[string]))
    
    return tabs

def print_guitar_tabs(tabs):
    for string, frets in tabs.items():
        print(f"{string}|{''.join(frets)}|")


def main():
    duration = int(input("How many seconds of audio do you want to record?: "))

    audio_file = record_audio(duration)
    pitch_sequence, sr = extract_pitch(audio_file)
    notes = generate_notes(pitch_sequence, sr)
    
    print("\nDetected Notes:\n")
    for note, duration in notes:
        print(f"{note} ({duration:.2f}s)")

    print("\nGuitar Tabs:\n")
    tabs = generate_guitar_tabs(notes)
    print_guitar_tabs(tabs)

if __name__ == "__main__":
    main()