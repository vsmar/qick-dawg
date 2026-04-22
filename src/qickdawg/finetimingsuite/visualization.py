'''
Visualization utilities for Fine Timing Suite sequences
=======================================================================
Basic plotting functions for pulse sequences and results.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class PulseSequenceVisualizer:
    """
    Minimal visualization utilities for Fine Timing Suite pulse sequences.
    """

    @staticmethod
    def plot_rabi_sequence(cfg=None):
        """
        Plot Rabi oscillation pulse sequence with overlay of configuration.

        Parameters
        ----------
        cfg : NVConfiguration or None
            Configuration object. If None, shows generic labels.
        """
        graphics_folder = os.path.join(os.path.dirname(__file__), 'graphics')
        image_path = os.path.join(graphics_folder, 'RABI.png')

        if not os.path.exists(image_path):
            plt.text(0.5, 0.5, 'RABI.png not found', ha='center', va='center')
            return

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(mpimg.imread(image_path))

        if cfg is None:
            plt.text(455, 510, "config.reps", fontsize=14)
            plt.text(350, 440, "config.laser_on", fontsize=14)
            plt.text(265, 355, "config.readout_integration", fontsize=14)
            plt.text(527, 355, "config.readout_integration", fontsize=14)
            plt.text(735, 355, "config.relax_delay", fontsize=14)
            plt.text(220, 407, "config.laser_readout_offset", fontsize=14)
        else:
            plt.text(420, 510, f"Repeat {cfg.reps} times", fontsize=14)
            plt.text(350, 440, f"laser_on_tus = {str(cfg.laser_on_tus)[:4]} us", fontsize=14)
            plt.text(265, 370, f"readout_integration = {int(cfg.readout_integration_tns)} ns", fontsize=14)
            plt.text(527, 370, f"readout_integration = {int(cfg.readout_integration_tns)} ns", fontsize=14)
            plt.text(735, 370, f"relax_delay = {int(cfg.relax_delay_tns)} ns", fontsize=14)
            plt.text(235, 407, f"laser_offset = {int(cfg.laser_readout_offset_tns)} ns", fontsize=14)

        plt.title("Rabi Oscillation Pulse Sequence", fontsize=20)

    @staticmethod
    def plot_cpmg_sequence(cfg=None):
        """
        Plot CPMG-XY pulse sequence with overlay of configuration.

        Parameters
        ----------
        cfg : NVConfiguration or None
            Configuration object. If None, shows generic labels.
        """
        graphics_folder = os.path.join(os.path.dirname(__file__), 'graphics')
        image_path = os.path.join(graphics_folder, 'CPMG.png')

        if not os.path.exists(image_path):
            plt.text(0.5, 0.5, 'CPMG.png not found. Showing text summary.', ha='center', va='center')
            if cfg is not None:
                plt.text(0.5, 0.4, f"CPMG repeats: {cfg.n_cpmg}", ha='center', va='center')
                plt.text(0.5, 0.3, f"Tau sweep: {cfg.tau_start_ftsamp} to {cfg.tau_end_ftsamp} ftsamp", ha='center', va='center')
            return

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(mpimg.imread(image_path))

        if cfg is None:
            plt.text(450, 510, "config.reps", fontsize=14)
            plt.text(450, 450, "config.n_cpmg", fontsize=14)
            plt.text(450, 390, "config.tau (swept)", fontsize=14)
        else:
            plt.text(420, 510, f"Repeat {cfg.reps} times", fontsize=14)
            plt.text(420, 450, f"N_CPMG = {cfg.n_cpmg}", fontsize=14)
            plt.text(420, 390, f"Tau: {cfg.tau_start_ftsamp} to {cfg.tau_end_ftsamp} ftsamp", fontsize=14)

        plt.title("CPMG-XY Pulse Sequence", fontsize=20)

    @staticmethod
    def plot_ramsey_sequence(cfg=None):
        """
        Plot Ramsey pulse sequence with overlay of configuration.

        Parameters
        ----------
        cfg : NVConfiguration or None
            Configuration object. If None, shows generic labels.
        """
        graphics_folder = os.path.join(os.path.dirname(__file__), 'graphics')
        image_path = os.path.join(graphics_folder, 'RAMSEY.png')

        plt.figure(figsize=(12, 10))
        plt.axis('off')

        if os.path.exists(image_path):
            plt.imshow(mpimg.imread(image_path))
        else:
            plt.text(0.5, 0.5, 'π/2 — τ (swept) — π/2 — Readout', ha='center', va='center', fontsize=16)

        if cfg is None:
            plt.text(0.5, 0.1, "Tau sweep: config.tau_start to config.tau_end", ha='center', fontsize=12)
        else:
            plt.text(0.5, 0.1, f"Tau sweep: {cfg.tau_start_ftsamp} to {cfg.tau_end_ftsamp} ftsamp", 
                     ha='center', fontsize=12)

        plt.title("Ramsey Pulse Sequence (T2* measurement)", fontsize=16)

    @staticmethod
    def plot_t1_sequence(cfg=None):
        """
        Plot T1 pulse sequence with overlay of configuration.

        Parameters
        ----------
        cfg : NVConfiguration or None
            Configuration object. If None, shows generic labels.
        """
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.6, 'π — τ (swept) — Readout', ha='center', va='center', fontsize=16)

        if cfg is None:
            plt.text(0.5, 0.3, "Delay sweep: config.t1_delay_start to config.t1_delay_end", 
                     ha='center', fontsize=12)
        else:
            plt.text(0.5, 0.3, f"Delay sweep: {cfg.t1_delay_start_treg} to {cfg.t1_delay_end_treg} treg", 
                     ha='center', fontsize=12)

        plt.title("T1 Pulse Sequence (Spin Relaxation)", fontsize=16)

    @staticmethod
    def plot_podmr_sequence(cfg=None):
        """
        Plot PODMR (pulsed ODMR) pulse sequence with frequency sweep overlay.

        Parameters
        ----------
        cfg : NVConfiguration or None
            Configuration object. If None, shows generic labels.
        """
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.6, 'π (swept frequency) — Readout', ha='center', va='center', fontsize=16)

        if cfg is None:
            plt.text(0.5, 0.3, "Frequency sweep: config.mw_start_fMHz to config.mw_end_fMHz", 
                     ha='center', fontsize=12)
        else:
            plt.text(0.5, 0.3, f"Frequency sweep: {cfg.mw_start_fMHz} to {cfg.mw_end_fMHz} MHz", 
                     ha='center', fontsize=12)

        plt.title("PODMR Pulse Sequence (Frequency Sweep)", fontsize=16)
