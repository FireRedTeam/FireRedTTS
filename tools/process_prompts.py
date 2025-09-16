import os
import argparse
import numpy as np
import glob


from pydub.silence import split_on_silence
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def find_audio_files(folder_path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(
            glob.glob(os.path.join(folder_path, "**", f"*{suffix}"), recursive=True)
        )
    return files


def slice(audio_path):
    """_summary_

    Args:
        audio_path (_type_): audio path
    """
    try:
        audio = AudioSegment.from_file(audio_path)
    except:
        print(audio_path)
        return 0

    segments = split_on_silence(
        audio, min_silence_len=200, silence_thresh=-50, seek_step=100, keep_silence=100
    )

    print("---segments:\n", segments)

    combined = segments[0]
    for i in range(1, len(segments)):
        combined += segments[i]
    return combined


def process(root_dir, out_dir):
    """_summary_

    Args:
        root_dir (_type_): input audio dir
        out_dir (_type_): output audio dir
    """
    audio_files = find_audio_files(root_dir, suffixes=[".wav"])
    print("---audio_files:\n", audio_files, len(audio_files))

    for audio_path in audio_files:
        print("---audio_path:", audio_path)
        combined = slice(audio_path=audio_path)
        name = audio_path.split(sep="/")[-1]
        combined.export(os.path.join(out_dir, name), format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    assert os.path.exists(args.root_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    process(root_dir=args.root_dir, out_dir=args.out_dir)
