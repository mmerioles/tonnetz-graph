# Tonnetz Graph

## Overview

## Datasets

## Project Timeline

1. **Week 1 Goals - Monophonic Tonnetz Graph**
   - [ ] Parse midi, and build adjacency matrix
   - [ ] Given adjacency matrix, make visualization of graph
   - [ ] given adjacency matrix, find centrality, betweenness, eigenvector, etc
   - [ ] Given adjacency matrix, find degree distribution, clustering coefficient, etc

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

## Usage
To launch, please run your desired scripts using uv
```bash
uv run main.py
```

## Team Members
- Shreeya Ajay Bhonsle
- Aditi Pagey
- Nishant Arya
- Matthew Merioles

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

