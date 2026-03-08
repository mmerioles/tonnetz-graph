# Tonnetz Graph

## Overview

## Datasets

## Project Timeline

1. **Week 1 Goals - Monophonic Tonnetz Graph**
   - [ ] Parse midi, and build adjacency matrix
   - [x] Given adjacency matrix, make visualization of graph
   - [x] Given adjacency matrix, find centralities: degree, betweenness, eigenvector
   - [x] Given adjacency matrix, find statistics: degree distributions, clustering coefficient, diameter, giant component

2. **Week 2 Goals - TBD**


## Installation
First, clone the repository via
```bash
git clone https://github.com/mmerioles/tonnetz-graph.git
```
Next, please install uv if you do not have this. Used for dependency and version management
```bash
# For mac/linux
curl -LsSf https://astral.sh/uv/install.sh | sh 

# For windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 
```
While in the project directory, run the following to sync your environment with needed dependencies
```bash
uv sync
```
For audio playback features, you need to install FluidSynth: https://www.fluidsynth.org/

Note: These instructions are for windows

Open powershell as administrator and run the following to install chocolatey
```bash
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
After installing chocolatey, install fluid synth
```bash
choco install fluidsynth
```
## Usage
To run a test, please run your desired scripts using uv
```bash
uv run tests/test_centrality.py # Debug print out centralities
uv run tests/test_statistics.py # Debug print out statistics
uv run tests/test_plot.py # Debug print graph
```
To smoke check against pytest, simply run the following
```bash
uv run pytest
```

## Team Members
- Shreeya Ajay Bhonsle
- Aditi Pagey
- Nishant Arya
- Matthew Merioles

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

