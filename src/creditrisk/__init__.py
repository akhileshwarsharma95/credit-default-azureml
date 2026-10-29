"""Shared library for the credit default (PD) solution.

Everything that must behave identically in training, batch scoring, the online
endpoint and the Streamlit app lives here, so there is one implementation of
features, explanations, business policy and monitoring logic.
"""

__version__ = "1.0.0"
