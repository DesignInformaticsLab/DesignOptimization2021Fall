Solve the following problem using [Python SciPy.optimize][]. Please attach your code and
results. Specify your initial guesses of the solution. If you change
your initial guess, do you find different solutions? (**100 points**)

$$\\begin{aligned}
                &\\text{minimize:} && (x_1-x_2)^2 + (x_2+x_3-2)^2 + (x_4-1)^2+(x_5-1)^2 \\\\
                &\\text{subject to:} && x_1 + 3x_2 = 0 \\\\
                &&& x_3 + x_4 - 2x_5 = 0 \\\\
                &&& x_2 - x_5 = 0 \\\\
                &&& -10 \\leq x_i \\leq 10, \~i=1,\\ldots,5
              \\end{aligned}$$

**Note**:

1.  Please learn how to use **break points** to debug. **I will not
    address your programming questions if you have not learned how to
    debug your code.**

2.  I recommend [PyCharm][] as the IDE. If you are new to Python, you can also start with [Google Colab][] without installing anything.
    
3.  If you are on Windows, the [Anaconda][] version of Python 3 is highly recommended.


**Here are the steps to push a homework submission**:

1.  Clone the [course repo][]: First click on **Code** to get the
 Git address (e.g., the HTTPS address). Then use your IDE to clone (download) the repo using this address. 
 [PyCharm tutorial on using Git][].

2.  You will find the homework in the **Homework** folder.

3.  For analytical problems (e.g., proofs and calculations), please use [Markdown][] to type up your answers. 
[Markdown Math][]. For Latex users, you can convert tex to markdown using [Pandoc][]. 

4. For coding problems, please submit a [Jupyter Notebook][] file with your code and final results. 
You can also add a URL to your Jupyter or Colab Notebook in README.md if you use online notebooks.

5. For each homework, please submit a single notebook file (or link) that combines the markdown solutions, 
the codes, and the computation results, and name the file according to the homework.  

6. **IMPORTANT** Please push (upload) the notebook file every time you work on the 
homework and add comments when you push, e.g., "finished problem 1, still debugging problem 2". This way I 
know you worked on your own.
 

[Python SciPy.optimize]: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#
[PyCharm]: https://www.jetbrains.com/pycharm/promo/?utm_source=bing&utm_medium=cpc&utm_campaign=AMER_en_US-PST%2BMST_PyCharm_Branded&utm_term=pycharm&utm_content=pycharm
[Google Colab]: https://colab.research.google.com
[Anaconda]: https://anaconda.org/anaconda/python
[course repo]: https://github.com/DesignInformaticsLab/DesignOptimization2021Fall
[PyCharm tutorial]: https://www.jetbrains.com/help/pycharm/set-up-a-git-repository.html#clone-repo
[Pandoc]: https://pandoc.org/try/
[Jupyter Notebook]: https://jupyter.org/try
[Markdown]: https://guides.github.com/features/mastering-markdown/
[Markdown Math]: http://luvxuan.top/posts/Markdown-math/