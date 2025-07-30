"""
Utility functions for the scripts in the visualisation folder.
"""


# Helper function to better manage size in matplotlib
def cm2inch(x: float) -> float:
    """
    Convert centimeters to inches.
    """
    return x / 2.54
