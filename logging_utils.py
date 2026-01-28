# coding: utf-8
"""
Common logging and CSV output utilities for EIT simulations
"""

import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Logger:
    """
    Logger that outputs to both console and file simultaneously
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Path to log file. Auto-generated if None
        """
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"simulation_log_{timestamp}.txt"

        self.log_file = Path(log_file)
        self.terminal = sys.stdout

        # Initialize log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Simulation Log ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

    def write(self, message: str):
        """Output message to both terminal and file"""
        self.terminal.write(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message)

    def flush(self):
        """Flush buffer"""
        self.terminal.flush()


class CSVWriter:
    """
    Save simulation results to CSV files
    """

    def __init__(self, base_filename: str):
        """
        Args:
            base_filename: Base name for CSV files (without extension)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_filename = base_filename
        self.timestamp = timestamp

    def save_position_errors(
        self,
        days: List[int],
        contrasts: List[float],
        results: Dict[str, Any],
        resistance_data: Optional[Dict[int, int]] = None,
        filename: Optional[str] = None,
    ):
        """
        Save position error data to CSV

        Args:
            days: List of days
            contrasts: List of contrasts
            results: Results dict (key: noise level, value: error list)
            resistance_data: Resistance data dict (optional)
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            filename = f"{self.base_filename}_position_errors_{self.timestamp}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header row
            header = ["Day", "Contrast"]
            if resistance_data is not None:
                header.insert(1, "Resistance_Ohm")

            noise_levels = sorted(results.keys())
            for noise in noise_levels:
                header.append(f"PE_Noise_{noise * 100:.1f}%")

            writer.writerow(header)

            # Data rows
            for i, day in enumerate(days):
                row = [day]
                if resistance_data is not None:
                    row.append(resistance_data.get(day, "N/A"))
                row.append(f"{contrasts[i]:.4f}")

                for noise in noise_levels:
                    if isinstance(results[noise], dict):
                        # Statistical version (with mean, std)
                        if "mean" in results[noise]:
                            pe = results[noise]["mean"][i]
                        else:
                            pe = results[noise][i] if i < len(results[noise]) else None
                    else:
                        # Simple version
                        pe = results[noise][i] if i < len(results[noise]) else None

                    row.append(f"{pe:.6f}" if pe is not None and pe == pe else "NaN")

                writer.writerow(row)

        print(f"CSV saved: {filename}")

    def save_statistical_results(
        self,
        days: List[int],
        contrasts: List[float],
        results: Dict[str, Dict[str, List[float]]],
        filename: Optional[str] = None,
    ):
        """
        Save statistical results (mean, std) to CSV

        Args:
            days: List of days
            contrasts: List of contrasts
            results: Results dict {noise_level: {"mean": [...], "std": [...]}}
            filename: Output filename
        """
        if filename is None:
            filename = f"{self.base_filename}_statistical_{self.timestamp}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header row
            header = ["Day", "Contrast"]
            noise_levels = sorted(results.keys())
            for noise in noise_levels:
                header.extend(
                    [
                        f"Mean_Noise_{noise * 100:.1f}%",
                        f"Std_Noise_{noise * 100:.1f}%",
                    ]
                )

            writer.writerow(header)

            # Data rows
            for i, day in enumerate(days):
                row = [day, f"{contrasts[i]:.4f}"]
                for noise in noise_levels:
                    mean_val = results[noise]["mean"][i]
                    std_val = results[noise]["std"][i]

                    row.append(f"{mean_val:.6f}" if mean_val == mean_val else "NaN")
                    row.append(f"{std_val:.6f}" if std_val == std_val else "NaN")

                writer.writerow(row)

        print(f"CSV saved: {filename}")

    def save_comparison_results(
        self,
        days: List[int],
        contrasts: List[float],
        results_greit: Dict[str, List[float]],
        results_jac: Dict[str, List[float]],
        filename: Optional[str] = None,
    ):
        """
        Save GREIT vs JAC comparison results to CSV

        Args:
            days: List of days
            contrasts: List of contrasts
            results_greit: GREIT results {"mean": [...], "std": [...]}
            results_jac: JAC results {"mean": [...], "std": [...]}
            filename: Output filename
        """
        if filename is None:
            filename = f"{self.base_filename}_comparison_{self.timestamp}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header row
            header = [
                "Day",
                "Contrast",
                "GREIT_Mean",
                "GREIT_Std",
                "JAC_Mean",
                "JAC_Std",
                "Difference(JAC-GREIT)",
                "Better_Method",
            ]
            writer.writerow(header)

            # Data rows
            for i, day in enumerate(days):
                mean_g = results_greit["mean"][i]
                std_g = results_greit["std"][i]
                mean_j = results_jac["mean"][i]
                std_j = results_jac["std"][i]

                # Calculate difference
                if mean_g == mean_g and mean_j == mean_j:  # Not NaN
                    diff = mean_j - mean_g
                    better = "GREIT" if diff > 0 else "JAC"
                else:
                    diff = float("nan")
                    better = "N/A"

                row = [
                    day,
                    f"{contrasts[i]:.4f}",
                    f"{mean_g:.6f}" if mean_g == mean_g else "NaN",
                    f"{std_g:.6f}" if std_g == std_g else "NaN",
                    f"{mean_j:.6f}" if mean_j == mean_j else "NaN",
                    f"{std_j:.6f}" if std_j == std_j else "NaN",
                    f"{diff:+.6f}" if diff == diff else "NaN",
                    better,
                ]
                writer.writerow(row)

        print(f"CSV saved: {filename}")


def setup_logging(script_name: str) -> Logger:
    """
    Setup logging and return Logger object

    Args:
        script_name: Script name (without extension)

    Returns:
        Logger object
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{script_name}_log_{timestamp}.txt"
    logger = Logger(log_file)

    # Replace sys.stdout
    sys.stdout = logger

    print(f"Logging to: {log_file}")
    print()

    return logger


def finalize_logging(logger: Logger):
    """
    Finalize logging

    Args:
        logger: Logger object
    """
    print("\n" + "=" * 70)
    print(f"Log saved to: {logger.log_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Restore sys.stdout
    sys.stdout = logger.terminal
