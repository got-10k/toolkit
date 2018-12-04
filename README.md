# GOT-10k Python Toolkit

This repository contains the official python toolkit for running experiments and evaluate performance on [GOT-10k](https://got-10k.github.io) benchmark. For convenience, it also provides unofficial implementation of tracking pipelines for [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html) and [VOT](http://votchallenge.net) benchmarks. The code is written in pure python and is compile-free. Although we support both python2 and python3, we recommend python3 for better performance.

[GOT-10k](https://got-10k.github.io) is a large, high-diversity and one-shot database for evaluating generic purposed visual trackers. If you use the GOT-10k database or toolkits for a research publication, please consider citing:

```Bibtex
"GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild."
L. Huang, X. Zhao and K. Huang,
arXiv:1810.11981, 2018.
```

&emsp;\[[Project](https://got-10k.github.io)\]\[[PDF](https://arxiv.org/abs/1810.11981)\]\[[Bibtex](https://got-10k.github.io/bibtex)\]

## Table of Contents

* [Installation](#installation)
* [Quick Start: A Concise Example](#quick-start-a-concise-example)
* [Quick Start: Jupyter Notebook for Off-the-Shelf Usage](#quick-start-jupyter-notebook-for-off-the-shelf-usage)
* [How to Define a Tracker?](#how-to-define-a-tracker)
* [How to Run Experiments on GOT-10k?](#how-to-run-experiments-on-got-10k)
* [How to Evaluate Performance?](#how-to-evaluate-performance)
* [How to Loop Over GOT-10k Dataset?](#how-to-loop-over-got-10k-dataset)
* [Issues](#issues)

### Installation

Install the toolkit using `pip` (recommended):

```bash
pip install git+https://github.com/got-10k/toolkit.git@master
```

Or, alternatively, clone the repository and install dependencies:

```
git clone https://github.com/got-10k/toolkit.git
cd toolkit
pip install -r requirements.txt
```

Then directly copy `got10k` folder to your workspace to use it.

### Quick Start: A Concise Example

Here is a simple example on how to use the toolkit to define a tracker, run experiments on GOT-10k and evaluate performance.

```Python
from got10k.trackers import BaseTracker
from got10k.experiments import ExperimentGOT10k

class IdentityTracker(BaseTracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box

if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('data/GOT-10k', subset='val')
    experiment.run(tracker, visualize=True)

    # report performance
    experiment.report([tracker.name])
```

To run experiments on [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html) or [VOT](http://votchallenge.net) benchmarks, simply change `ExperimentGOT10k` to `ExperimentOTB` or `ExperimentVOT`, and `root_dir` to their corresponding paths for this purpose.

### Quick Start: Jupyter Notebook for Off-the-Shelf Usage

Open [quick_examples.ipynb](https://github.com/got-10k/toolkit/tree/master/examples/quick_examples.ipynb) in [Jupyter Notebook](http://jupyter.org/) to see more examples on toolkit usage.

### How to Define a Tracker?

To define a tracker using the toolkit, simply inherit and override `init` and `update` methods from the `BaseTracker` class. Here is a simple example:

```Python
from got10k.trackers import BaseTracker

class IdentityTracker(BaseTracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',  # tracker name
            is_deterministic=True    # stochastic (False) or deterministic (True)
        )
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box
```

### How to Run Experiments on GOT-10k?

Instantiate an `ExperimentGOT10k` object, and leave all experiment pipelines to its `run` method:

```Python
from got10k.experiments import ExperimentGOT10k

# ... tracker definition ...

# instantiate a tracker
tracker = IdentityTracker()

# setup experiment (validation subset)
experiment = ExperimentGOT10k(
    root_dir='data/GOT-10k',    # GOT-10k's root directory
    subset='val',               # 'train' | 'val' | 'test'
    result_dir='results',       # where to store tracking results
    report_dir='reports'        # where to store evaluation reports
)
experiment.run(tracker, visualize=True)
```

The tracking results will be stored in `result_dir`.

### How to Evaluate Performance?

Use the `report` method of `ExperimentGOT10k` for this purpose:

```Python
# ... run experiments on GOT-10k ...

# report tracking performance
experiment.report([tracker.name])
```

When evaluated on the __validation subset__, the scores and curves will be directly generated in `report_dir`.

However, when evaluated on the __test subset__, since all groundtruths are withholded, you will have to submit your results to the [evaluation server](https://got-10k.github.io/submit_instructions) for evaluation. The `report` function will generate a `.zip` file which can be directly uploaded for submission. For more instructions, see [submission instruction](https://got-10k.github.io/submit_instructions).

### How to Loop Over GOT-10k Dataset?

The `got10k.datasets.GOT10k` provides an iterable and indexable interface for GOT-10k's sequences. Here is an example:

```Python
from PIL import Image
from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame

dataset = GOT10k(root_dir='data/GOT-10k', subset='train')

# indexing
img_file, anno = dataset[10]

# for-loop
for s, (img_files, anno) in enumerate(dataset):
    seq_name = dataset.seq_names[s]
    print('Sequence:', seq_name)

    # show all frames
    for f, img_file in enumerate(img_files):
        image = Image.open(img_file)
        show_frame(image, anno[f, :])
```

To loop over `OTB` or `VOT` datasets, simply change `GOT10k` to `OTB` or `VOT` for this purpose.

### Issues

Please report any problems or suggessions in the [Issues](https://github.com/got-10k/toolkit/issues) page.
