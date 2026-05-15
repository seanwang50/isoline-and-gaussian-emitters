# IsoLine and Gaussian Emitters

Directional variation outperforms random mutation on a relatively simple quality-diversity search.

## Run script
```bash
#!/bin/bash
python3 -m venv .venv

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi

pip install --upgrade pip
pip install -r setup.txt
python script.py
```

## Cite paper
```bibtex
@misc{wang2026isoline,
  title={IsoLine and Gaussian Emitters},
  author={Wang, Sean},
  month={may},
  year={2026}
}
```
