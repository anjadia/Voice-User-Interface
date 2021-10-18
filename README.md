# Voice-User-Interface
Voice-User Interface

```
~/VOI/Voice-User-Interface ❯ tree
├── asr
│   ├── asr.py
│   ├── __init__.py
│   ├── README.md
│   ├── test
│   ├── tools
│   │   └── __init__.py
│   └── train
├── README.md
├── requirements.txt
├── tts
│   ├── __init__.py
│   ├── README.md
│   ├── test
│   ├── tools
│   │   └── __init__.py
│   ├── train
│   └── tts.py
└── voice_user_interface
    ├── run.py
    └── tools
```

# TODO

- resampling 
- normalizer 

## ASR:

`in <repo_root>/asr`

- MFCC 
- train xvector 
- train/valid set

## TTS: 

`in <repo_root>/tts`

- diphones? 

## Interface

`in <repo_root>/voice_user_interface`

- asr + tts to user inference


