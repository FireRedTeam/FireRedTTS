import os
import json
import argparse
import torchaudio

from fireredtts.fireredtts_redserving import FireRedTTS_Redserving


def test(text_path, tts_model, speaker, out_dir):
    """_summary_

    Args:
        text_path (_type_): _description_
    """
    os.makedirs(out_dir, exist_ok=True)
    f_in = open(file=text_path, encoding="utf-8")
    lines = f_in.readlines()
    print("########### Test Sentence: ############\n", lines, len(lines), "\n")

    i = 1
    for text in lines:
        wrong_code, out_wav = tts_model.synthesize_online(
            speaker_id=speaker, text=text, lang="auto"
        )
        if wrong_code is None:
            torchaudio.save(os.path.join(out_dir, str(i) + ".wav"), out_wav, 24000)

        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", default="")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--out_dir", default="")

    args = parser.parse_args()
    assert os.path.exists(args.text_path)

    tts_redserving = FireRedTTS_Redserving(
        model_config_path="configs/model_config.json",
        speaker_config_path="configs/speaker_config.json",
        pretrained_path=os.path.join(os.path.dirname(__file__), "pretrained_models"),
    )

    speaker_config = json.load(open("configs/speaker_config.json"))
    print("---speaker_config:\n", speaker_config)
    speaker_list = list(speaker_config.keys())

    if args.speaker == "all":
        for speaker in speaker_list:
            print("---speaker:", speaker)
            out_dir = os.path.join(args.out_dir, speaker)
            test(
                text_path=args.text_path,
                tts_model=tts_redserving,
                speaker=speaker,
                out_dir=out_dir,
            )
    else:
        out_dir = os.path.join(args.out_dir, args.speaker)
        test(
            text_path=args.text_path,
            tts_model=tts_redserving,
            speaker=args.speaker,
            out_dir=out_dir,
        )
