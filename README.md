# snippets
useful, reusable python, tf, etc code snippets; settings and preferences


## Jupyter-related notes

[auto-reload][auto_reload_instructions] edited modules.

For better code review, export jupyter notebook code to .py file:
```
!jupyter-nbconvert notebook-name.ipynb --to python --PythonExporter.exclude_input_prompt=True
```
[jupytext][jupytext] to do it automatically

### Settings for plotting

* in Jupyter notebook: `%matplotlib notebook`
allows interactive figures if your environment allows it.

    Retina specific, also set in each jupyter notebook:
    ```bash
    %config InlineBackend.figure_format = 'retina'
    ```
    
    Or add the following line to your ipython_kernel_config.py, which for me is in ~/.ipython/profile_default/
    
    ```bash
    c.IPKernelApp.matplotlib = 'notebook'
    c.InlineBackend.figure_format = 'retina'
    ```
    
    If the file does not already exist, you can generate it with all settings commented out by entering ipython profile create at the command line.

* in Jupyter lab:
  better still to use `%matplotlib widget` in jupyter lab. 
  Change defaults in config file to `widget` or **do not set** a default. In lab
  cannot set the plot-mode twice.
  [Install widgets.][for_widgets]


[auto_reload_instructions]: https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html
[for_widgets]: https://github.com/matplotlib/jupyter-matplotlib
[jupytext]: https://github.com/mwouts/jupytext

## Python Environments

I don't use conda. Personal choice?

* pyenv, virtualenv, venv, [pipenv][pipenv]? Too many options: [explained][py_envs]

* [tox][tox] in turn uses pyenv
  <details><summary>Example script with multiple environments:</summary>

    ```
    [tox]
    envlist = py36, flake8, coverage, research
    skipsdist = True
    sitepackages = True
    
    [testenv]
    setenv = VIRTUAL_ENV={envdir}
             SKIP_GENERATE_AUTHORS=1
    usedevelop = True
    install_command =
        pip install {opts} {packages}
    deps =
        -r{toxinidir}/requirements.txt
        -r{toxinidir}/test-requirements.txt
    whitelist_externals = sh
    
    
    [testenv:py36]
    basepython = python3.6
    setenv = {[testenv]setenv}
    commands =
        pytest -v {posargs} -k tests/
    deps =
        {[testenv]deps}
    
    
    [testenv:research]
    basepython = python3.6
    setenv = {[testenv]setenv}
    commands =
        pytest -v {posargs} -k tests/
    deps =
        {[testenv]deps}
        -r{toxinidir}/research-requirements.txt
    
   
    [flake8]
    max-line-length = 119
    
    [testenv:flake8]
    basepython = python3.6
    usedevelop = False
    deps =
        flake8
    commands=
        flake8 $CODE_FOLDER(s)
    
    
    [testenv:coverage]
    basepython = python3.6
    setenv = {[testenv]setenv}
    commands =
        pytest -v --cov=$CODE_FOLDER --cov-report term-missing -k tests/
    deps =
        {[testenv]deps}
    ```
    </summary>
   
 
[tox]: https://tox.readthedocs.io/en/latest/
[pipenv]: https://docs.python-guide.org/dev/virtualenvs/
[py_envs]: https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe


## Python Profiling

```bash
python -m cProfile your_program.py
python -m pstats profile
OR, PREFERRED:
snakeviz profile
```

* **ncalls**,     the number of calls. 

* **tottime**,    the total time spent in the given function (and excluding time made in
               calls to sub-functions) 

* **percall**    is the quotient of tottime divided by ncalls 

* **cumtime** is the cumulative time spent in this and all subfunctions (from invocation
                till exit). This figure is accurate even for recursive functions. 

* **percall**    is the quotient of cumtime divided by primitive calls 

## Useful linux

* powerful remote and local file transfer:
  ```
  rsync --info=progress2 from_path to_path
  ``` 
  [progress explained][rsync_prog2]. I always use rsync for backing up
  large/many files, even (esp) locally on Mac.

[rsync_prog2]: https://unix.stackexchange.com/questions/215271/understanding-the-output-of-info-progress2-from-rsync