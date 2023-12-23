# RVC NgNgNgan

python script to download & process data to train a R.V.C. model of M.C. Nguyễn Ngọc Ngạn

tải và xử lí audio (tách giọng và xoá khoảng lặng) để train R.V.C. nhái giọng bác Ngạn

vì lí do bản quyền nên ở đây chỉ có code ko có data ko có model, ai muốn thì đọc hướng dẫn dưới đây để chạy code kéo audio về tự train

![license](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

## requirements

need NVIDIA GPU

install `ffmpeg`

`git clone` this repo

prepare a fresh python env (`venv` or `conda`)
```bash
pip install -r requirements.txt
```
see https://github.com/pyannote/pyannote-audio to accept ToS of https://huggingface.co/pyannote

then create huggingface token then login with
```bash
huggingface-cli login --token=███
```

## data preparation

get list of audios and info (speakers count): `python scripts/00-get-info.py`

small test with 2 audio files: `yt-dlp "KgxWziSHQP8" "01WRW7IV1uQ" -x --audio-format "wav" -o "%(id)s.%(ext)s" -P "data/raw"`

if good to go then download all youtube audios: `python scripts/01-download-audio.py`

if u want simplicity, download only video with 1 speaker (uncomment the code in file `scripts/01-download-audio.py`), then in following steps u only need to remove silence (i.e. skip diarization + voice isolation + merge) so directly to train R.V.C.

audios are saved as `.wav` files in folder `data/1-raw`

remove silence: `python scripts/02-remove-silence.py`

audios are saved as `.wav` files in folder `data/2-vad`

speaker diarization: `python scripts/03-diarization.py`

audios are cut per speaker in folder `data/3-diarized` → listen carefully to each segment (speakers count sometimes not reliable) then remove segments not Nguyễn Ngọc Ngạn

do diarization before voice isolation because `demucs` on large audio raise out-of-memory errors

(seem useless / no effect) ~~isolate voices from music/sound effects : `python scripts/04-isolate-voice.py`, audios saved in folder `data/4-voices`~~

merge segments together: `python scripts/05-merge-segments.py`

audios saved in folder `data/5-merged`

## e.g. of R.V.C. training + inference

using https://github.com/IAHispano/Applio-RVC-Fork

6h30min audio at 48 kHz + RMVPE pitch extraction = 16.1 GiB disk space

pretrained base models (Discriminator & Generator) v2

hop length only relevant to “crepe” algorithm

train feature index (independent of actual model training)

how to resume training from previous ckpt: increase number of epochs

max batch size depends on VRAM: e.g. 6 if 6 GiB VRAM

during training, monitor 2 losses: G total & D total, save ckpt every 5 epochs so can stop early before overfitting

train 120 epochs but keep ckpt at 80th epoch, see [loss curve](tensorboard/plot_tensorboard.ipynb)

save model to share:
- save voice for inference only
- save D & G so other can resume training

inference parameters:
- volume envelope = 0
- protect voiceless consonants = 0.5 to disable
- search feature ratio: 3 intervals to test: <0.4, 0.4-0.7, >0.7
- autotune more relevant if singing
- median filtering only relevant to “harvest” algorithm
