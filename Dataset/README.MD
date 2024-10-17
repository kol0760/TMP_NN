The original data for this paper is hosted on [Zenodo](https://zenodo.org/records/13906515?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI2M2I1OGExLWQwYjEtNDA0Mi1hMjM2LWY0NTBlOWQ1MmI0NyIsImRhdGEiOnt9LCJyYW5kb20iOiJlMWQ2Mzc5ZmJkNWNhNGEwNzBlZjVkM2M1YmE0Yzc5NCJ9.5kaxbMcJXoCIr0A7c5oGVOPKKU7nbEG7MMFqOGtS1W8B4OLxVyCXjyCyB2tcdrDwTllz4aTdBpEa_BzE-hLNvw), and has been packaged in pkl format for training and testing. You can use Python's `pickle` library to read this data. The sample code to do so is as follows:

```python
import pickle

with open('Bergman_cyclization.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
```

This code will open and read the data stored in the `Bergman_cyclization.pkl` file, loading it into the `data` variable. You can then process or manipulate the data depending on its specific structure.