## Setup
``` shell
conda deactivate
conda activate webshop
```

### Setup WebShop Environment
1. `cd webshop`
2. Follows instruction in `webshop/readme.md`
*The installation scripts were modified so that it works on the author's machine setup. If failed, try installing from the [original repository](https://github.com/princeton-nlp/WebShop).*

### Setup ROIL
1. Update .env
```
PYTHONPATH=$PYTHONPATH:$PWD/webshop
OPENAI_APIKEY=your-api-key
```

2. install requirement
``` shell
chmod +X setup.sh
. ./setup.sh
pip install -r requirement
```

## Running
### Rule optimization
To run rule optimization steps, use `runner/webshop/train.py`. The main function of the script has hyperparameters described in the paper.

### Evaluation
Use `runner/webshop/eval.py` to evaluate an optimized ruleset. Hyperparameters are either inherited from the optimization log or being the default values.

### Alignment
The alignment experiment discussed in the paper was done using `runner/webshop/align.py`.