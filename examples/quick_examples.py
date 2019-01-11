from __future__ import absolute_import, print_function

import fire
from PIL import Image

from got10k.trackers import Tracker, IdentityTracker
from got10k.experiments import ExperimentGOT10k
from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame


ROOT_DIR = 'data/GOT-10k'


def example_track_val_set():
    # setup tracker
    tracker = IdentityTracker()

    # run experiment on validation set
    experiment = ExperimentGOT10k(
        root_dir=ROOT_DIR,
        subset='val',
        result_dir='results',
        report_dir='reports')
    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])


def example_track_test_set():
    # setup tracker
    tracker = IdentityTracker()

    # run experiment on test set
    experiment = ExperimentGOT10k(
        root_dir=ROOT_DIR,
        subset='test',
        result_dir='results',
        report_dir='reports')
    experiment.run(tracker, visualize=False)

    # a ".zip" file will be generated ready for submission
    # follow the guide to submit your results to
    # http://got-10k.aitestunion.com/
    experiment.report([tracker.name])


def example_plot_curves():
    # reports of 25 baseline entries can be downloaded from
    # http://got-10k.aitestunion.com/downloads
    report_files = [
        'reports/GOT-10k/performance_25_entries.json']
    tracker_names = [
        'SiamFCv2', 'GOTURN', 'CCOT', 'MDNet']
    
    # setup experiment and plot curves
    experiment = ExperimentGOT10k('data/GOT-10k', subset='test')
    experiment.plot_curves(report_files, tracker_names)


def example_loop_dataset():
    # setup dataset
    dataset = GOT10k(ROOT_DIR, subset='val')

    # loop over the complete dataset
    for s, (img_files, anno) in enumerate(dataset):
        seq_name = dataset.seq_names[s]
        print('Sequence:', seq_name)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            box = anno[f, :]  # (left, top, width, height)
            show_frame(image, box, colors='w')


def example_show():
    # setup experiment
    experiment = ExperimentGOT10k(
        root_dir=ROOT_DIR,
        subset='test',
        result_dir='results',
        report_dir='reports')

    # visualize tracking results
    tracker_names = [
        'SiamFCv2', 'GOTURN', 'CCOT', 'MDNet']
    experiment.show(tracker_names)


if __name__ == '__main__':
    # choose an example function to execute, e.g.,
    # > python quick_examples example_loop_dataset
    fire.Fire()
