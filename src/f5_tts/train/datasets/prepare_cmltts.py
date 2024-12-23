import csv
import shutil
from datasets import load_dataset
import os
import soundfile as sf

destination_path = "/home/sidleal/sources/F5-TTS-ptbr/data/cmltts"

def process_cmltts():
    seen_chars = set()
    os.makedirs(f"{destination_path}", exist_ok=True )
    fm = open(destination_path+ "/metadata.csv", 'w')
    fm.write("audio_file|text\n")
    mls = load_dataset("ylacombe/cml-tts", "portuguese", split="train+dev+test", streaming=False)
    #mls = load_dataset("ylacombe/cml-tts", "portuguese", split="dev", streaming=False)

    batch_size = 1000
    for i, batch in enumerate(mls.iter(batch_size=batch_size)):
        print(f"Processing batch {i+1}")
        for j in range(0, len(batch['text'])):
            #print(batch['text'][j])
            text = batch['text'][j]
            if not text:
                continue
            audio = batch['audio'][j]['array']
            path = batch['audio'][j]['path']
            sample_rate = batch['audio'][j]['sampling_rate']
            #print(f"Text: {text} Audio: {audio} sample_rate: {sample_rate} path: {path}")
            fm.write(f"wavs/{path}|{text}\n")
            os.makedirs(f"{destination_path}/wavs", exist_ok=True )
            filename = os.path.join(f"{destination_path}/wavs", path)
            sf.write(filename, audio, sample_rate)
            for char in text:
                seen_chars.add(char)
            print(f"Saved audio to: {filename}")
    
    sorted_list = sorted(list(seen_chars))
    print(sorted_list)
    with open(destination_path + "/vocab.txt", "w", encoding="utf-8") as f:
        for item in sorted_list:
            f.write(item + "\n")

    fm.close()

process_cmltts()