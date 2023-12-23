#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""remove silence using Silero VAD"""
# see https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies
# also https://github.com/snakers4/silero-vad/blob/master/utils_vad.py

import os
from tqdm import tqdm
import torch
import torchaudio

from _constants import LIST_VID, RAW_DATA, VAD_DATA

TQDM_PBAR_FORM = "{percentage: 5.1f}% |{bar}| {n:.0f}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
SAMPLING_RATE = 16000  # Silero VAD operating value
MODEL, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(
	repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
)
MODEL = MODEL.to("cuda")


def vad_filter(infile: str, outfile: str) -> None:
	wav = read_audio(infile, sampling_rate=SAMPLING_RATE).to("cuda")  # SileroVAD operate on mono channel at 16 kHz
	with tqdm(total=wav.shape[0], bar_format=TQDM_PBAR_FORM) as pbar:
		speech_timestamps = get_speech_timestamps(
			wav, MODEL, sampling_rate=SAMPLING_RATE,
			progress_tracking_callback=lambda val: pbar.update(val * 10)  # weird, TODO: raise issue in silero repo
		)
	# speech_timestamps is list[dict[str, int]]

	# convert timestamps to match original audio file (dual channels & higher bit rate)
	waveform, sample_rate = torchaudio.load(infile)
	metadata = torchaudio.info(infile)
	ratio = sample_rate / SAMPLING_RATE
	cut_waveform = torch.cat([
		waveform[:, int(el["start"] * ratio) : int(el["end"] * ratio)]
		for el in speech_timestamps
	], dim=1)
	torchaudio.save(outfile, cut_waveform, sample_rate, bits_per_sample=metadata.bits_per_sample, encoding=metadata.encoding)
	# torch.cuda.empty_cache()


#################################### main #####################################
for id in LIST_VID.keys():
	infile = os.path.join(RAW_DATA, f"{id}.wav")
	outfile = os.path.join(VAD_DATA, f"{id}.wav")
	if not os.path.exists(infile):
		print(f"{id} not found")
	else:
		print(f"{id} to be VAD filtered")
		vad_filter(infile, outfile)
