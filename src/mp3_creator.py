import numpy as np
from pydub import AudioSegment
import numpy as np

def generate_tone(frequency, duration_ms, volume=-20.0):
    sample_rate = 44100  # Samples per second
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)  # Generate sine wave
    tone = (tone * (2**15 - 1)).astype(np.int16)  # Convert to 16-bit PCM
    audio = AudioSegment(
        tone.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    return audio + volume  # Adjust volume

def generate_frequency_map():
    keys = ["A_0", "A#_0", "B_0", "C_1", "C#_1", "D_1", "D#_1", "E_1", "F_1", "F#_1", "G_1", "G#_1",
            "A_1", "A#_1", "B_1", "C_2", "C#_2", "D_2", "D#_2", "E_2", "F_2", "F#_2", "G_2", "G#_2",
            "A_2", "A#_2", "B_2", "C_3", "C#_3", "D_3", "D#_3", "E_3", "F_3", "F#_3", "G_3", "G#_3",
            "A_3", "A#_3", "B_3", "C_4", "C#_4", "D_4", "D#_4", "E_4", "F_4", "F#_4", "G_4", "G#_4",
            "A_4", "A#_4", "B_4", "C_5", "C#_5", "D_5", "D#_5", "E_5", "F_5", "F#_5", "G_5", "G#_5",
            "A_5", "A#_5", "B_5", "C_6", "C#_6", "D_6", "D#_6", "E_6", "F_6", "F#_6", "G_6", "G#_6",
            "A_6", "A#_6", "B_6", "C_7", "C#_7", "D_7", "D#_7", "E_7", "F_7", "F#_7", "G_7", "G#_7",
            "A_7", "A#_7", "B_7", "C_8"]
    base_freq = 440.0  # Frequency of A4
    freq_map = {key: base_freq * (2 ** ((i - 48) / 12)) for i, key in enumerate(keys)}
    return freq_map


def combine_notes(keys, freq_map, duration_ms=500):
    combined = AudioSegment.silent(duration=duration_ms)
    for key in keys:
        if key in freq_map:
            tone = generate_tone(freq_map[key], duration_ms)
            combined = combined.overlay(tone)
    return combined


def create_mp3(steps, output_file="output.mp3", step_duration_ms=500):
    freq_map = generate_frequency_map()
    song = AudioSegment.silent(duration=0)

    for step_keys in steps:
        step_audio = combine_notes(step_keys, freq_map, duration_ms=step_duration_ms)
        song += step_audio

    song.export(output_file, format="mp3")


def main():
    steps = [
        {"C4", "E4", "G4"},  # Step 1: C major chord
        {"D4", "F4", "A4"},  # Step 2: D minor chord
        {"E4", "G4", "B4"}  # Step 3: E minor chord
    ]

    create_mp3(steps, "my_music.mp3", step_duration_ms=500)


if __name__ == "__main__":
    main()
