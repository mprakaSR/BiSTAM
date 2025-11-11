"""
dataloader.py

Unified dataloader for EEG 2025 challenges.
Supports:
  - Challenge 1: Contrast Change Detection (reaction-time regression)
  - Challenge 2: Externalizing-factor regression (controlled per-task sampling)
"""

from pathlib import Path
import math
import random
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_windows_from_events, create_fixed_length_windows,
)
from braindecode.datasets.base import BaseConcatDataset, BaseDataset, EEGWindowsDataset
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# Import configuration
from config import (
    DATA_DIR, SFREQ, TASKS,
    EXCLUDE_SUBJECTS, BATCH_SIZE, NUM_WORKERS,
)

import random

def random_offset(max_shift_s=0.05):
    """Random ±shift around start time in seconds."""
    return random.uniform(-max_shift_s, max_shift_s)
    
class EEGChallengeDataLoader:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.sfreq = SFREQ
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.sub_rm = EXCLUDE_SUBJECTS

    # ------------------------------------------------------------------
    # CHALLENGE 1: Contrast Change Detection (Reaction Time Prediction)
    # ------------------------------------------------------------------
    def load_challenge1(self,releases ):
        """
        Prepare dataloaders for Challenge 1:
        Contrast-change-detection task (reaction-time regression).
        Loads from all configured releases (skips missing tasks).
        """
        from config import (
           SFREQ, BATCH_SIZE, NUM_WORKERS, EXCLUDE_SUBJECTS
        )

        print("📥 Loading Challenge 1 dataset from releases:", releases)

        # --------------------------------------------------------------
        # Load EEG datasets from all available releases
        # --------------------------------------------------------------
        datasets = []
        for release in releases:
            try:
                ds = EEGChallengeDataset(
                    release=release,
                    task="contrastChangeDetection",
                    cache_dir=self.data_dir,
                    mini=False,
                    download=False,
                )
                datasets.append(ds)
                print(f"✅ Loaded release {release}")
            except Exception as e:
                print(f"⚠️ Skipping release {release}: {e}")

        if len(datasets) == 0:
            raise RuntimeError("❌ No valid releases found for contrastChangeDetection!")

        dataset_ccd = BaseConcatDataset(datasets)
        print(f"📦 Combined {len(dataset_ccd.datasets)} datasets for Challenge 1.")

        # --------------------------------------------------------------
        # Offline preprocessing
        # --------------------------------------------------------------
        EPOCH_LEN_S = 2.0
        ANCHOR = "stimulus_anchor"
        SHIFT_AFTER_STIM_S = 0.5
        WINDOW_LEN_S = 2.0

        transformation_offline = [
            Preprocessor(
                annotate_trials_with_target,
                target_field="rt_from_stimulus",
                epoch_length=EPOCH_LEN_S,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]

        preprocess(dataset_ccd, transformation_offline, n_jobs=-1)

        dataset = keep_only_recordings_with("stimulus_anchor", dataset_ccd)

        # --------------------------------------------------------------
        # Create stimulus-locked windows
        # --------------------------------------------------------------
        jitter = random_offset()
        single_windows = create_windows_from_events(
            dataset,
            mapping={"stimulus_anchor": 0},
            
            trial_start_offset_samples=int(SHIFT_AFTER_STIM_S  * SFREQ),
            trial_stop_offset_samples=int((SHIFT_AFTER_STIM_S + WINDOW_LEN_S) * SFREQ),
            window_size_samples=int(EPOCH_LEN_S * SFREQ),
            window_stride_samples=SFREQ,
            preload=True,
        )

        single_windows = add_extras_columns(
            single_windows,
            dataset,
            desc="stimulus_anchor",
            keys=("target", "rt_from_stimulus", "stimulus_onset", "response_onset"),
        )

        # --------------------------------------------------------------
        # Split by subject (train/val/test)
        # --------------------------------------------------------------
        meta = single_windows.get_metadata()
        subjects = [s for s in meta["subject"].unique() if s not in EXCLUDE_SUBJECTS]

        VALID_FRAC = 0.1
        TEST_FRAC = 0.1
        RANDOM_SEED = 2025

        train_sub, valid_test = train_test_split(
            subjects,
            test_size=(VALID_FRAC + TEST_FRAC),
            random_state=check_random_state(RANDOM_SEED),
            shuffle=True,
        )
        valid_sub, test_sub = train_test_split(
            valid_test,
            test_size=TEST_FRAC,
            random_state=check_random_state(RANDOM_SEED + 1),
            shuffle=True,
        )

        subject_split = single_windows.split("subject")
        train_ds = BaseConcatDataset([subject_split[s] for s in train_sub])
        valid_ds = BaseConcatDataset([subject_split[s] for s in valid_sub])
        test_ds = BaseConcatDataset([subject_split[s] for s in test_sub])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        print(f"✅ Challenge 1 ready: Train={len(train_ds)}, Val={len(valid_ds)}, Test={len(test_ds)}")
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # CHALLENGE 2: Externalizing Factor Regression (Controlled per-task sampling)
    # ------------------------------------------------------------------
    def load_challenge2(self,releases, task_fraction: float = 0.25):
        """
        Prepare dataloaders for Challenge 2:
        Predict the externalizing factor from EEG recordings.

        Loads multiple tasks per release, but keeps only a fraction of
        each task (e.g., 60% of contrastChangeDetection, 60% of RestingState).

        Args:
            task_fraction (float): Fraction [0–1] of each task to keep per release.
        """
        from config import (
            TASKS, SFREQ, BATCH_SIZE, NUM_WORKERS, EXCLUDE_SUBJECTS
        )

        print(f"📥 Loading Challenge 2 dataset from releases: {releases}")
        print(f"🎯 Including tasks: {TASKS}, using {task_fraction * 100:.0f}% per task per release")

        rng = check_random_state(2025)
        all_datasets = []

        # --------------------------------------------------------------
        # Load datasets for all releases × tasks
        # --------------------------------------------------------------
        for release in releases:
            for task in TASKS:
                try:
                    ds = EEGChallengeDataset(
                        release=release,
                        task=task,
                        mini=False,
                        download=False,
                        description_fields=[
                            "subject", "session", "run", "task", "age",
                            "gender", "sex", "externalizing"
                        ],
                        cache_dir=self.data_dir,
                    )
                    n_total = len(ds.datasets)
                    n_keep = max(1, int(task_fraction * n_total))
                    selected = rng.choice(range(n_total), n_keep, replace=False)
                    kept = [ds.datasets[i] for i in selected]
                    print(f"✅ Release={release}, Task={task}: kept {n_keep}/{n_total}")
                    all_datasets.extend(kept)
                except Exception as e:
                    print(f"⚠️ Skipping release={release}, task={task}: {e}")

        if not all_datasets:
            raise RuntimeError("❌ No valid Challenge 2 datasets found!")

        all_datasets = BaseConcatDataset(all_datasets)
        print(f"📦 Combined {len(all_datasets.datasets)} total recordings after sampling.")

        # --------------------------------------------------------------
        # Filter unusable subjects
        # --------------------------------------------------------------
        filtered = [
            ds for ds in all_datasets.datasets
            if (
                ds.description["subject"] not in EXCLUDE_SUBJECTS
                and ds.raw.n_times >= 4 * SFREQ
                and len(ds.raw.ch_names) == 129
                and not math.isnan(ds.description["externalizing"])
            )
        ]
        all_datasets = BaseConcatDataset(filtered)
        print(f"✅ Remaining after filtering: {len(all_datasets.datasets)}")

        # --------------------------------------------------------------
        # Subject mapping
        # --------------------------------------------------------------
        subjects = sorted(set(
            ds.description["subject"]
            for ds in all_datasets.datasets
            if ds.description["subject"] not in EXCLUDE_SUBJECTS
        ))
        subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
        self.subject_to_idx = subject_to_idx
        print(f"🧠 Created mapping for {len(subject_to_idx)} subjects.")

        # --------------------------------------------------------------
        # Split subjects into train/val/test
        # --------------------------------------------------------------
        VALID_FRAC, TEST_FRAC, RANDOM_SEED = 0.1, 0.1, 2025
        train_sub, valid_test = train_test_split(
            subjects,
            test_size=(VALID_FRAC + TEST_FRAC),
            random_state=check_random_state(RANDOM_SEED),
            shuffle=True,
        )
        valid_sub, test_sub = train_test_split(
            valid_test,
            test_size=TEST_FRAC,
            random_state=check_random_state(RANDOM_SEED + 1),
            shuffle=True,
        )

        # --------------------------------------------------------------
        # Create 4-second windows (stride = 2 s)
        # --------------------------------------------------------------
        windows_ds = create_fixed_length_windows(
            all_datasets,
            window_size_samples=4 * SFREQ,
            window_stride_samples=2 * SFREQ,
            drop_last_window=True,
        )

        # --------------------------------------------------------------
        # Wrapper dataset for 2-s crop and metadata
        # --------------------------------------------------------------
        class ExtDatasetWrapper(BaseDataset):
            def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int,
                         subject_to_idx: dict, seed=None):
                self.dataset = dataset
                self.crop_size_samples = crop_size_samples
                self.subject_to_idx = subject_to_idx
                self.rng = random.Random(seed)

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                X, _, crop_inds = self.dataset[idx]
                desc = self.dataset.description
                target = float(desc["externalizing"])
                subj = desc["subject"]
                subj_id = self.subject_to_idx.get(subj, -1)

                infos = {
                    "subject": subj,
                    "subj_id": subj_id,
                    "sex": desc["sex"],
                    "age": float(desc["age"]),
                    "task": desc["task"],
                }

                i_window_in_trial, i_start, i_stop = crop_inds
                start_offset = self.rng.randint(0, max(0, i_stop - i_start - self.crop_size_samples))
                X = X[:, start_offset : start_offset + self.crop_size_samples]
                return X, target, (i_window_in_trial, i_start, i_stop), infos

        # --------------------------------------------------------------
        # Split by subject and wrap
        # --------------------------------------------------------------
        subject_split = windows_ds.split("subject")
        train_ds = BaseConcatDataset([subject_split[s] for s in train_sub])
        valid_ds = BaseConcatDataset([subject_split[s] for s in valid_sub])
        test_ds = BaseConcatDataset([subject_split[s] for s in test_sub])

        train_ds = BaseConcatDataset([
            ExtDatasetWrapper(ds, crop_size_samples=2 * SFREQ, subject_to_idx=subject_to_idx)
            for ds in train_ds.datasets
        ])
        valid_ds = BaseConcatDataset([
            ExtDatasetWrapper(ds, crop_size_samples=2 * SFREQ, subject_to_idx=subject_to_idx)
            for ds in valid_ds.datasets
        ])
        test_ds = BaseConcatDataset([
            ExtDatasetWrapper(ds, crop_size_samples=2 * SFREQ, subject_to_idx=subject_to_idx)
            for ds in test_ds.datasets
        ])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        print(f"✅ Challenge 2 ready:")
        print(f"   Train: {len(train_ds)} windows ({len(train_sub)} subjects)")
        print(f"   Val:   {len(valid_ds)} windows ({len(valid_sub)} subjects)")
        print(f"   Test:  {len(test_ds)} windows ({len(test_sub)} subjects)")
        print(f"Each sample shape: [batch, 129, {2 * SFREQ}]")

        return train_loader, val_loader, test_loader

    def load_reference(self, challenge: int, release, **kwargs):
        """
        Load data for either EEG Challenge 1 or 2, using only the specified release.

        Args:
            challenge (int): Challenge number (1 or 2).
            release (int): EEG-2025 release to load (default: 2).
            **kwargs: Additional keyword arguments passed to the specific loader.

        Returns:
            (DataLoader, DataLoader, DataLoader): train, validation, and test dataloaders.
        """
        
        

        if challenge == 1:
            print(f"🔄 Loading reference dataset for Challenge 1 (Release {release})")
            return self.load_challenge1(releases=release, **kwargs)

        elif challenge == 2:
            print(f"🔄 Loading reference dataset for Challenge 2 (Release {release})")
            return self.load_challenge2(releases=release, **kwargs)

        else:
            raise ValueError(f"❌ Invalid challenge number: {challenge}. Must be 1 or 2.")

# ==========================================================
# Example run
# ==========================================================
if __name__ == "__main__":
    loader = EEGChallengeDataLoader()

    # Challenge 2 with 60% of each task per release
    train_loader, val_loader, test_loader = loader.load_challenge2(releases=["R5"],task_fraction=0.25)
    X, y, crop_inds, infos = next(iter(train_loader))
    print(f"Batch loaded successfully: X={X.shape}, y={y.shape}, ")
