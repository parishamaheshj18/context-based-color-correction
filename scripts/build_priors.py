"""Build reflectance priors from data/reflectance_spectra/<class>/ (same as `ncc build-priors`)."""

from natural_color_charts.cli import app

if __name__ == "__main__":
    app(["build-priors"])
