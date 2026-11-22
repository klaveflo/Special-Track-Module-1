# Special-Track-Module-1
In this repository all tasks for the special-track module 1 of the Master's program in Applied Computational Life Sciences are solved.

## Setup  

1. Create an environment with Python 3.12 or newer and install the dependencies 

```bash
conda create -n movie-agent python=3.12
conda activate movie-agent
pip install -r requirements.txt
```

2. Install [Ollama](https://ollama.com/) and pull a model (currently the workflow is being tested with `llama3.1:8b` and `llama3.2`)

```bash
ollama pull llama3.1:b
```

3. Download MovieLens latest-small from https://grouplens.org/datasets/movielens/ and unzip it to `data/ml-latest-small/`. Run `notebooks/03_movie_data.ipynb` to create `data/cleaned_movie_data.csv` which serves as the cleaned dataset for the following notebooks.


## Structure
```text
Special-Track-Module-1
|
├── data/               # MovieLens download, cleaned catalog, local preference file
|
├── notebooks/          # One jupyter notebook per week solving the relevant tasks
|
├── src/                # Shared python modules used by the notebooks
  ├── catalog.py        # Loads the catalog once, shared helpers
  ├── tools.py          # Catalog tools: search, filter, rank, details
  ├── memory.py         # Local feedback store (likes and dislikes)
  └── retrieval.py      # Tag-based TF-IDF retrieval: search_descriptions()
|
├── outputs/            # Generated results
|
└── requirements.txt    # Textfile with the required python libraries
```

---

## Dataset Citation
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872