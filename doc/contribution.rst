Contributing to inne
=====================

Hi! Thanks for your interest in contributing to [inne](inne - inne 0.0.1 documentation) :D .
In this document we'll try to summarize everything that you need to know to do a good job.


Code and Issues
---------------

We use `Github <https://github.com/xhan97/inne>`_ to host our code repositories
and issues.You can look at `issues <https://github.com/xhan97/inne/issues>`_ to report any
issues related to pgmpy. Here is a `guide <https://guides.github.com/features/issues/>`_
on how to report better issues.

Git and our Branching model
---------------------------

Git
---

We use `Git <http://git-scm.com/>`_ as our `version control
system <http://en.wikipedia.org/wiki/Revision_control>`_, so the best way to
contribute is to learn how to use it and put your changes on a Git repository.
There are plenty of documentation about Git -- you can start with the `Pro Git
book <http://git-scm.com/book/>`_.
Or You can go through the `try git tutorial <https://try.github.io/levels/1/challenges/>`_.

Forks + GitHub Pull requests
----------------------------

We use the famous
`gitflow <http://nvie.com/posts/a-successful-git-branching-model/>`_ to manage our
branches.

Summary of our git branching model:

- Fork the desired repository on GitHub to your account;
- Clone your forked repository locally: `git clone git@github.com:your-username:repository-name.git`;
- Create a new branch off of `develop` with a descriptive name (for example: `feature/portuguese-sentiment-analysis`, `hotfix/bug-on-downloader`). You can do it by switching to `develop` branch (`git checkout develop`) and then creating a new branch (`git checkout -b name-of-the-new-branch`);
- Do many small commits on that branch locally (`git add files-changed`, `git commit -m "Add some change"`);
- Push to your fork on GitHub (with the name as your local branch: `git push origin branch-name`);
- Create a pull request using the GitHub Web interface (asking us to pull the changes from your new branch and add the changes to our `develop` branch);
- Wait for comments.


Tips
----

- Write `helpful commit messages <http://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message>`_.
- Anything in the `dev` branch should be deployable (no failing tests).
- Never use `git add .`: it can add unwanted files;
- Avoid using `git commit -a` unless you know what you're doing;
- Check every change with `git diff` before adding then to the index (stage area) and with `git diff --cached` before commiting;
- If you have push access to the main repository, please do not commit directly to `dev`: your access should be used only to accept pull requests; if you want to make a new feature, you should use the same process as other developers so that your code can be reviewed.


Documentation Guidelines
------------------------




Code Guidelines
---------------

- We use `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_;
- We permit 120 characters in a line, rather 79 as suggested in PEP8
- Write tests for your new features (please see "Tests" topic below);
- Always remember that `commented code is dead code <http://www.codinghorror.com/blog/2008/07/coding-without-comments.html>`_;
- Name identifiers (variables, classes, functions, module names) with readable names (`x` is always wrong);
- When manipulating strings, use `Python's new-style formatting <http://docs.python.org/library/string.html#format-string-syntax>`_ (`'{} = {}'.format(a, b)` instead of `'%s = %s' % (a, b)`);
- When working with files use `with open(<filename>, <option>) as f` instead of ` f = open(<filename>, <option>)`;
- Run all tests before pushing (`pytest`) so you will know if your changes broke something;


Tests
-----

We use `Github Actions <https://github.com/features/actions>`_ for continous integration
and python `pytest <https://docs.pytest.org/en/stable/index.html>`_ for writing tests.
You should write tests for every feature you add or bug you solve in the code.
Having automated tests for every line of our code let us make big changes
without worries: there will always be tests to verify if the changes introduced
bugs or lack of features. If we don't have tests we will be blind and every
change will come with some fear of possibly breaking something.

For a better design of your code, we recommend using a technique called
`test-driven development <https://en.wikipedia.org/wiki/Test-driven_development>`_,
where you write your tests **before** writing the actual code that implements
the desired feature.


Discussion
----------

Please feel free to contact us through mail list if
you have any questions or suggestions.
Every contribution is very welcome!

Happy hacking! ;)