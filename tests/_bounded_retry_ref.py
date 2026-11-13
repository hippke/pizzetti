"""emcee reference for test_marginal.test_fit_is_bounded_retry (the test reads the stored result,
testdata/bounded_retry_ref.json, instead of running emcee each time):

    python tests/_bounded_retry_ref.py

The model is test_marginal._bounded_retry_model() (30-min cadence, 1-h transit, a/R* pressed against its upper
bound); the reference is fit(method="emcee", seed=1, n=4000) on the marginal posterior: median and standard deviation
of rp, a and b. The posterior does not depend on the sampler, so the reference stays valid when the importance
sampler changes; rerun this if the model or the likelihood changes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "testdata", "bounded_retry_ref.json")

if __name__ == "__main__":
    from test_marginal import _bounded_retry_model
    m = _bounded_retry_model()
    e = m.fit(method="emcee", seed=1, n=4000)
    out = {q: dict(median=float(np.median(e.samples[:, e.labels.index(q)])), std=float(np.std(e.samples[:, e.labels.index(q)])))
           for q in ("rp", "a", "b")}
    out["n_samples"] = int(len(e.samples))
    with open(PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
