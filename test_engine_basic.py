import pandas as pd
import importlib
import sys, os

# Allow running tests without installing the package
here = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, "src"))

from statilytics_studio.core import Engine

def test_engine_describe_returns_dataframe():
    df = pd.DataFrame({"a":[1,2,3,4], "b":[5,6,7,8]})
    e = Engine(df)
    desc = e.describe()
    assert hasattr(desc, "shape")
    assert "missing" in desc.columns
