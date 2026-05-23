#!/bin/bash
# Download example audio if not present
if [ ! -f "audio.wav" ]; then
    wget -O audio.wav "https://www.openslr.org/resources/12/test-clean-wav/61/70968-0000.wav"
fi
echo "Hello world" > transcript.txt

# Run alignment
python ../align.py audio.wav transcript.txt --output alignment.json --srt