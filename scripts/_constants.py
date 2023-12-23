# -*- coding: utf-8 -*-

import os.path
import json

RAW_DATA = os.path.join("data", "1-raw")
VAD_DATA = os.path.join("data", "2-vad")
DIARIZED_DATA = os.path.join("data", "3-diarized")
VOICE_DATA = os.path.join("data", "4-voices")
MERGED_DATA = os.path.join("data", "5-merged")

DRAFT_FILE = os.path.join("data", "draft.json")
SUMMARY_FILE = os.path.join("data", "data.json")
with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
	LIST_VID = json.load(f)
