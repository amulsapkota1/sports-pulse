import json, re, ast, math, orjson
import pandas as pd
import numpy as np

# Robust JSON loader (handles "", None, malformed)
def safe_json_load(x):
    if pd.isna(x) or x=="":
        return {}
    # handle double-quoted JSON that might already be a dict-like string
    try:
        return orjson.loads(x)
    except:
        try:
            return json.loads(x)
        except:
            try:
                return ast.literal_eval(x)
            except:
                return {}

def to_seconds_mmss(s):
    if pd.isna(s) or s=="":
        return np.nan
    if isinstance(s, (int,float)):
        return float(s)
    m = re.match(r"^(\d{1,2}):(\d{2})$", str(s).strip())
    if not m:
        return np.nan
    return int(m.group(1))*60 + int(m.group(2))

def coalesce(*vals):
    for v in vals:
        if v not in (None, "", np.nan) and not (isinstance(v,float) and math.isnan(v)):
            return v
    return np.nan