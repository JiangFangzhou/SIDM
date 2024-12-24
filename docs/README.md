## Notes by Yangyao

In order to publish our codes to Python Package Index, we have to reorganize our
repository, as the following layouts:

```text
repo_root/
    README.md
    pyproject.toml            # Project metadata
    LICENSE
    requirements.txt          # dependency list
    src/
        sidm/
           All source codes.
    docs/
        Jupyter notebooks.
        Demonstration files.
        Other Markdown files.
    tests/
        all test_ files.
```





## For developers

Yangyao has put his implementation under `src/sidm/yangyao/`. 
This is only for a reference, and shall be completely refactored and merged 
into the main implementation.

Some examples can be found under `docs/examples_yangyao.ipynb`.

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -s tests
```
This is automatically run on push & pull request in Github.