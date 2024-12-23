Once you save and unzip the folder

Navigate to the project directory:
C:FolderPath\FinalProject   
Then change into the microsearch directory by
bash 
cd microsearch

Then Create a python virtual environment
bash
python -m venv venv

==activate the environment== you should still be in microsearch after creating the virtual environment
bash
venv\Scripts\activate

and install the package and the dependencies
bash
pip install .

after installation launch the app
bash
python -m app.app --data-path output.parquet

and if you navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) you'll be able to query the engine.


Here's how to add the test running instructions to your README:

```markdown
[Previous installation steps...]

# Running the App
After installation launch the app
```bash
python -m app.app --data-path output.parquet
```
Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to query the engine.

# Running Tests
To run the test suite (make sure you're still in the microsearch directory with venv activated):

1. Run all tests:
```bash
python tests/test_engine.py
```

2. Or use pytest for detailed test reporting:
```bash
pytest tests/test_engine.py -v
```
The test suite includes:
- 70 test documents
- 50+ test queries
- Evaluation metrics (Precision@10, Recall@10, MRR, NDCG)
- Tests for document expansion and query expansion
```
