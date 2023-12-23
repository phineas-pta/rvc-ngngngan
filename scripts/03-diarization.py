#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""speaker diarization with pyannote"""
# see https://github.com/pyannote/pyannote-audio/blob/main/README.md

import os
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from _constants import LIST_VID, VAD_DATA, DIARIZED_DATA

MODEL = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device("cuda"))


def load_audio(filepath: str) -> dict[str, int | str | torch.Tensor]:
	"""return the format required by pyannote pipeline"""
	waveform, sample_rate = torchaudio.load(filepath)
	metadata = torchaudio.info(filepath)
	return {
		"waveform": waveform,  # waveform.shape = [2 audio channels, audio length in sec × sampling rate]
		"sample_rate": sample_rate,  # 48 kHz for all audio downloaded
		"bits_per_sample": metadata.bits_per_sample,  # default 16 for wav
		"encoding": metadata.encoding,  # default "PCM_S" for wav
	}


def cut_equal(infile: str, outdir: str, duration=1800) -> None:  # 1800sec = 30min
	audio_file = load_audio(infile)
	step = int(duration * audio_file["sample_rate"])
	segs = torch.split(audio_file["waveform"], split_size_or_sections=step, dim=-1)
	for i, seg in enumerate(segs):
		outfilename = f"SEG_{i:03d}.wav"
		outfile = os.path.join(outdir, outfilename)
		print(outfilename)
		torchaudio.save(
			outfile, seg,
			sample_rate=audio_file["sample_rate"],
			bits_per_sample=audio_file["bits_per_sample"],
			encoding=audio_file["encoding"]
		)
	# torch.cuda.empty_cache()


def diarize(audio_dict: dict[str, int | str | torch.Tensor], speakers_count: int) -> list[dict[str, int | str]]:
	"""
	input = output of load_audio()
	output = list of segments (start, end, speaker id)
	"""
	with ProgressHook() as hook:
		diarization = MODEL(audio_dict, hook=hook, num_speakers=speakers_count)
	raw_seg_list = []
	for turn, _, speaker in diarization.itertracks(yield_label=True):
		# print(f"start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")
		raw_seg_list.append({"start": turn.start, "end": turn.end, "speaker": speaker})

	# join segments with same speaker
	seg_list = [raw_seg_list[0]]
	for tmp in raw_seg_list[1:]:
		if seg_list[-1]["speaker"] == tmp["speaker"]:
			seg_list[-1]["end"] = tmp["end"]
		else:
			seg_list.append(tmp)

	# remove segment with length < 1s for pratical reason
	return [el for el in seg_list if el["end"] - el["start"] > 1]


def cut_diarize(infile: str, outdir: str) -> None:
	audio_file = load_audio(infile)
	segments_list = diarize(audio_file, speakers_count=speakers_count)
	for i, segment in enumerate(segments_list):
		outfilename = f"SEG_{i:03d} - {segment['speaker']}.wav"
		outfile = os.path.join(outdir, outfilename)
		print(outfilename)
		start_frame = int(segment["start"] * audio_file["sample_rate"])
		end_frame = int(segment["end"] * audio_file["sample_rate"])
		cut_waveform = audio_file["waveform"][:, start_frame:end_frame]
		torchaudio.save(
			outfile, cut_waveform,
			sample_rate=audio_file["sample_rate"],
			bits_per_sample=audio_file["bits_per_sample"],
			encoding=audio_file["encoding"]
		)
	# torch.cuda.empty_cache()


#################################### main #####################################
for id in LIST_VID.keys():
	infile = os.path.join(VAD_DATA, f"{id}.wav")
	if not os.path.exists(infile):
		print(f"{id} not found")
	else:
		speakers_count = LIST_VID[id]["speakers_count"]
		outdir = os.path.join(DIARIZED_DATA, id)
		os.makedirs(outdir, exist_ok=True)
		audio_file = load_audio(infile)

		if speakers_count == 1:
			print(f"{id} only 1 speaker")
			# cut_equal(infile, outdir)  # disabled coz also disable demucs coz useless
		else:
			print(f"{id} to be diarized: {speakers_count} speakers")
			cut_diarize(infile, outdir)
