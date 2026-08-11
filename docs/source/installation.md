# Installation

## Release

The [latest release](https://github.com/scams-research/fourierror/releases/latest) of `fourierror` can be installed from [PyPI](https://pypi.org/project/fourierror/) with `pip`:

```console
$ pip install fourierror
```

## Bleeding-edge

Alternatively, the latest development build can be installed from [GitHub](https://github.com/scams-research/fourierror), which can be installed directly from GitHub with `pip`:

```console
$ pip install git+https://github.com/scams-research/fourierror.git
```

Note, that if you already have `fourierror` on your system, you may need to run `pip uninstall fourierror` first to ensure you get the latest version.

## Development 

If you are interesting in modifying the `fourierror` code, you should clone the git repository and install `fourierror` with the `dev` option in editable mode. 

```console
$ git clone https://github.com/scams-research/fourierror.git
$ cd fourierror
$ pip install -e '.[dev]'
```

To run the notebooks included in the `docs` directory of the GitHub repository, it is necessary that the `[docs]` installation is performed. 

```console
$ git clone https://github.com/scams-research/fourierror.git
$ cd fourierror
$ pip install -e '.[docs]'
```

The documentation can then be built from the `Makefile` as follows (note that `pandoc` needs to be installed, either on `conda`/`mamba` or by following the [online instructions](https://pandoc.org/installing.html)). 

```console
$ cd docs
$ make html 
```

For some more "developer friendly" documentation, we have used Deep Wiki to create [detailed documentation](https://deepwiki.com/scams-research/kinisi), including some very useful flow diagrams of the code construction. 
