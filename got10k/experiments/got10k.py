from __future__ import absolute_import, division

import os
import numpy as np
import glob
import ast
import json
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from ..datasets import GOT10k
from ..utils.metrics import rect_iou
from ..utils.viz import show_frame


class ExperimentGOT10k(object):
    r"""Experiment pipeline and evaluation toolkit for GOT-10k dataset.
    
    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir,
                 result_dir='results', report_dir='reports'):
        super(ExperimentGOT10k, self).__init__()
        self.dataset = GOT10k(root_dir, subset='test', return_meta=True)
        self.result_dir = os.path.join(result_dir, 'GOT-10k')
        self.report_dir = os.path.join(report_dir, 'GOT-10k')
        self.nbins_iou = 101
        self.repetitions = 3

    def run(self, tracker, visualize=False):
        print('Running tracker %s on GOT-10k...' % tracker.name)

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                    tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, seq_name,
                    '%s_%03d.txt' % (seq_name, r + 1))
                if os.path.exists(record_file):
                    print('  Found results, skipping', seq_name)
                    continue

                # tracking loop
                boxes, times = tracker.track(
                    img_files, anno[0, :], visualize=visualize)
                
                # record results
                self._record(record_file, boxes, times)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        # visible ratios of all sequences
        seq_names = self.dataset.seq_names
        covers = {s: self.dataset[s][2]['cover'][1:] for s in seq_names}
        
        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            ious = {}
            times = {}
            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, anno, meta) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_files = glob.glob(os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_[0-9]*.txt' % seq_name))
                if len(record_files) == 0:
                    raise Exception('Results for sequence %s not found.' % seq_name)

                # read results of all repetitions
                boxes = [np.loadtxt(f, delimiter=',') for f in record_files]
                assert all([b.shape == anno.shape for b in boxes])

                # calculate and stack all ious
                bound = ast.literal_eval(meta['resolution'])
                seq_ious = [rect_iou(b[1:], anno[1:], bound=bound) for b in boxes]
                # only consider valid frames where targets are visible
                seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
                seq_ious = np.concatenate(seq_ious)
                ious[seq_name] = seq_ious

                # stack all tracking times
                times[seq_name] = []
                time_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_time.txt' % seq_name)
                if os.path.exists(time_file):
                    seq_times = np.loadtxt(time_file, delimiter=',')
                    seq_times = seq_times[~np.isnan(seq_times)]
                    seq_times = seq_times[seq_times > 0]
                    if len(seq_times) > 0:
                        times[seq_name] = seq_times

                # store sequence-wise performance
                ao, sr, speed, _ = self._evaluate(seq_ious, seq_times)
                performance[name]['seq_wise'].update({seq_name: {
                    'ao': ao,
                    'sr': sr,
                    'speed_fps': speed,
                    'length': len(anno) - 1}})

            ious = np.concatenate(list(ious.values()))
            times = np.concatenate(list(times.values()))

            # store overall performance
            ao, sr, speed, succ_curve = self._evaluate(ious, times)
            performance[name].update({'overall': {
                'ao': ao,
                'sr': sr,
                'speed_fps': speed,
                'succ_curve': succ_curve.tolist()}})
        
        # save performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot success curves
        self._plot_curves(performance, report_dir)

        return performance

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))
        
        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))
            
            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')
            
            # loop over the sequence and display results
            img_files, anno, _ = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())
        
        return len(set(records)) == 1

    def _evaluate(self, ious, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        thr_iou = np.linspace(0, 1, 101)
        bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, speed_fps, succ_curve

    def _plot_curves(self, performance, report_dir):
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        succ_file = os.path.join(report_dir, 'success_plot.png')
        key = 'overall'

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]
        
        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))
        
        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
