# Digital Signal Processing - Theory and Computational Examples

[![Linting](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/lint_nb.yml/badge.svg?branch=master)](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/lint_nb.yml) 
[![Run Notebooks](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/run_nb.yml/badge.svg?branch=master)](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/run_nb.yml) 
[![Sphinx Built](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/notebook_ci.yml/badge.svg?branch=master)](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/notebook_ci.yml)

This repository collects didactically edited [Jupyter](https://jupyter.org/) notebooks that introduce basic concepts of [Digital Signal Processing](https://en.wikipedia.org/wiki/Digital_signal_processing). Please take a look at the [static version](http://nbviewer.ipython.org/github/spatialaudio/digital-signal-processing-lecture/blob/master/index.ipynb)
at first glance. The materials provide an introduction to the foundations of spectral analysis, random signals, quantization, and filtering. A series of computational examples and exercises written in [IPython 3](http://ipython.org/) accompany the theory.

![Digital signal processing chain](https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/introduction/DSP.png)

The notebooks constitute the lecture notes to the master's course [Digital Signal Processing](http://www.int.uni-rostock.de/Digitale-Signalverarbeitung.48.0.html) given by [Sascha Spors](http://www.int.uni-rostock.de/Staff-Info.23+B6JmNIYXNoPWUxOTliMTNjY2U2MDcyZjJiZTI0YTc4MmFkYTE5NjQzJnR4X2pwc3RhZmZfcGkxJTVCYmFja0lkJTVEPTMmdHhfanBzdGFmZl9waTElNUJzaG93VWlkJTVEPTExMQ__.0.html) at the University of Rostock, Germany. The contents are provided as [Open Educational Resource](https://de.wikipedia.org/wiki/Open_Educational_Resources), so feel free to fork, share, teach and learn.
You can give the project a [Star](https://github.com/spatialaudio//digital-signal-processing-lecture/stargazers) if you like it.


## Getting Started

The Jupyter notebooks are accessible in various ways

* Online as [static web pages](http://nbviewer.ipython.org/github/spatialaudio/digital-signal-processing-lecture/blob/master/index.ipynb)
* Online for [interactive usage](https://mybinder.org/v2/gh/spatialaudio/digital-signal-processing-lecture/master?filepath=index.ipynb) with [binder](https://mybinder.org/)
* Local for interactive usage on your computer

Other online services (e.g. [Google Colaboratory](https://colab.research.google.com),
[Microsoft Azure](https://azure.microsoft.com/), ...) also provide environments for the 
interactive execution of Jupyter notebooks.
Local execution on your computer requires a local Jupyter/IPython installation.
The [Anaconda distribution](https://www.continuum.io/downloads)  is considered a convenient starting point.
Then, you would have to [clone/download the notebooks from Github](http://github.com/spatialaudio/digital-signal-processing-lecture).
Use a [Git](http://git-scm.org/) client to clone the notebooks and start
your local Jupyter server. For manual installation under OS X/Linux please
refer to your packet manager.

## Concept and Contents

An understanding of the underlying mechanisms and the limitations of basic digital signal processing methods is essential for designing more complex algorithms, such as the recent contributions on indirect [detection of supermassive
black holes](https://en.wikipedia.org/wiki/Messier_87)
heavily relying on system identification and image processing.

The present notebooks cover fundamental aspects of digital signal processing.
A focus is laid on a detailed mathematical treatise.
Discussing the mathematical background is essential to understand the underlying principles more broadly.
The materials contain computational examples and exercises to
interpret the theoretical findings and foster understanding.
The examples are designed to be explored interactively.
Furthermore, an outlook on practical applications is given whenever possible.

The material covers the following topics 

* spectral analysis of deterministic signals
* random signals and linear-time invariant systems
* spectral estimation for random signals
* realization of non-recursive and recursive filters
* design of digital filters


## Usage and Contributing

The contents are provided as [Open Educational Resource](https://de.wikipedia.org/wiki/Open_Educational_Resources).
The text is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
, and the code of the IPython examples is under the [MIT license](https://opensource.org/licenses/MIT).
Feel free to use the entire collection, parts, or even single notebooks for your purposes.
I am curious on the usage of the provided resources, so feel free to drop a
line or report to [Sascha.Spors@uni-rostock.de](mailto:Sascha.Spors@uni-rostock.de).

Our long-term vision is to lay the grounds for a **community-driven concise and
reliable resource** covering all relevant aspects of digital signal processing revised
by research and engineering professionals.
We aim to link the strengths of good old-fashioned textbooks
and the interactive playground of computational environments.
Open Educational Resources, combined with open source tools (Jupyter, Python) and well-established tools for data literacy (git), provides the unique possibility for collaborative and well-maintained resources.
Jupyter is chosen due to its seamless text, math, and code integration. The contents are represented future proof, as a simple markdown layout allowing for conversion into many other formats (html, PDF, ...). The git version management system features tracking of the changes and authorship.

You are invited to contribute on different levels.
The lowest level is to provide feedback in terms of a
[Star](https://github.com/spatialaudio/digital-signal-processing-lecture/stargazers)
if you like the content.
Please consider reporting errors or suggestions for improvements as
[issues](https://github.com/spatialaudio/digital-signal-processing-lecture/issues).
We are always looking forward to new examples and exercises, and reformulated existing and novel sub-sections or sections.
Authorship of each considerable contribution is clearly stated.
One way of introducing reformulated and new material is to handle them as
a tracked [pull request](https://github.com/spatialaudio/digital-signal-processing-lecture/pulls).


## Build Status

The notebooks' computational examples are automatically built and checked for errors by continuous integration using github actions.

[![Linting](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/lint_nb.yml/badge.svg?branch=master)](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/lint_nb.yml) 
[![Run Notebooks](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/run_nb.yml/badge.svg?branch=master)](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/run_nb.yml) 
[![Sphinx Built](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/notebook_ci.yml/badge.svg?branch=master)](https://github.com/spatialaudio/digital-signal-processing-lecture/actions/workflows/notebook_ci.yml)
