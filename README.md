# optimal-explorer-dev
Developing code for optimal exploration in language models

guess and answer word lists taken from https://gist.github.com/scholtes/94f3c0303ba6a7768b47583aff36654d
This is to match the setting from https://auction-upload-files.s3.amazonaws.com/Wordle_Paper_Final.pdf, which details that the optimal solution is obtained from starting with SALET and has a regret of 3.421

To clone this repo, you must run the following (this is to support a frok of verl, which adds combo lock environment support.):
```bash
git clone --recurse-submodules https://github.com/jakob-bjorner/optimal-explorer-dev.git
```

```bash
git submodule update --init --recursive
```


```bash
pip install -e .
```