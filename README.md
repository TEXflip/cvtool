# cvtool

Personal handy CV tools

### Installation

clone the repository and run `pip install .` inside the project directory, cvtool will be installed in the currently active python environment

### Usage examples

```bash
# convert an image to png
python -m cvtool ./image.jpg --convert png

# convert all the image in the current directory
python -m cvtool . --convert bmp

# image rectification
python -m cvtool ./image.png --rectify
```
