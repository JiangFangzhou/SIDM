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

Run tests:
```bash
python -m pytest -s tests
```