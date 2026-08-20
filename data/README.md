# Data

Both files are the course-competition distribution of **ACSF1** (Appliance Consumption Signatures Fingerprint) from the UCR Time Series Classification Archive.

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 100 | 1461 | **No header.** Column 0 is the integer class label 0–9; columns 1–1460 are the series. |
| `test.csv` | 100 | 1461 | **First row is a junk header** (`Id,,,,…`) — skip it. Column 0 holds string IDs `ap1`–`ap100`; columns 1–1460 are the series. |

Load them like this:

```python
import pandas as pd

train = pd.read_csv("data/train.csv", header=None)
y = train.iloc[:, 0].astype(int).values
X = train.iloc[:, 1:].astype(float).values

test = pd.read_csv("data/test.csv", skiprows=1, header=None)
test_ids = test.iloc[:, 0].astype(str).values
X_test = test.iloc[:, 1:].astype(float).values
```

Passing `header=None` to `test.csv` without `skiprows=1` silently reads the junk row as data and shifts every prediction by one row.

## Properties

- 10 classes, perfectly balanced: 10 training samples each.
- 1460 timesteps per trace, z-normalized to zero mean and unit variance.
- Classes correspond to 10 appliance categories: mobile phones (via chargers), coffee machines, computer stations, fridges and freezers, Hi-Fi systems, CFL lamps, laptops (via chargers), microwaves, printers, and televisions.

**The mapping from integer label to appliance name is not published with the dataset.** Listing the categories in the order above is not evidence that class 0 is "mobile phones". This project therefore refers to classes by index (`Class 0`–`Class 9`) throughout, and describes physical behaviour rather than asserting names — for example, "Class 8 switches on and off in long blocks" instead of "Class 8 is the fridge".

## Citation

> Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C.-C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., & Keogh, E. (2019). *The UCR Time Series Archive.* IEEE/CAA Journal of Automatica Sinica, 6(6), 1293–1305.

Dataset page: <https://www.timeseriesclassification.com/description.php?Dataset=ACSF1>

The files are redistributed here for reproducibility, under the archive's terms for research use.
