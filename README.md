`pip install -r requirements.txt`

# Voice-User-Interface
Voice-User Interface

```
~/VOI/Voice-User-Interface ❯ tree
├── asr
│   ├── asr.py
│   ├── __init__.py
│   ├── README.md
│   └── tools
│       └── __init__.py
├── config.yaml
├── preprocesing
│   ├── preprocesing.py
│   └── test.wav
├── README.md
├── requirements.txt
├── tts
│   ├── __init__.py
│   ├── README.md
│   ├── tools
│   │   └── __init__.py
│   └── tts.py
└── voice_user_interface
    └── run.py
```

# TODO

- resampling DONE
- normalizer DONE

## ASR:

`in <repo_root>/asr`

- MFCC 
- train GMM

## TTS: 

`in <repo_root>/tts`

- LCP

## Interface

`in <repo_root>/voice_user_interface`

- asr + tts to user inference


