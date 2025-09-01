# RLFlappy

RLFlappy is a Flappy Bird clone built with Python and Pygame, designed for reinforcement learning experiments and gameplay.

## Features
- Classic Flappy Bird gameplay
- Modular code structure for easy modification
- Asset folder with sprites and sounds
- Score display at the end of each game

## Requirements
- Python 3.7+
- Pygame

Install dependencies:
```bash
pip install -r requirements.txt
```


## How to Run

Navigate to the `src` folder in your terminal:
```bash
cd src
```
Then run the game using:
```bash
python -m game
```

## Folder Structure
```
RLFlappy/
├── src/
│   └── game/
│       ├── game.py
│       ├── __main__.py
│       ├── bird.py
│       ├── pipe.py
│       ├── pipeManager.py
│       ├── gameObject.py
│       └── ...
├── flappy-bird-assets/
│   ├── sprites/
│   └── audio/
├── requirements.txt
└── README.md
```

## Controls
- **Space**, **W**, or **Up Arrow**: Flap
- **Mouse Left Click**: Flap

## License
See the LICENSE file for details.

## Credits
- Flappy Bird assets from open sources (see asset folder LICENSE)
