
# Linear Regression Project(in developing, not validable yet)

Welcome to linear regression project.  Linear regression is the basic of many machine learning algorithms. In this project, you will imply what you learn to solve a linear regression problem, without using any external libraries. 

# What to do
All tasks are listed in  `linear_regression_project.ipynb`，including coding and proving tasks.

**You're encouragd to submit problem even if you haven't finished all tasks. You should submit with spefici questions and explain what you have tried and why it doesn't work. Reviewers will guide you accordingly.**


# Unit test
You can (and should) use unit tests to ensure all your implementations meet requirements. You can find the following code after every coding task. 

`%run -i -e test.py LinearRegressionTestCase.test_...`

If there is an error in your implementation, the `AssertionError` will be thrown. Please modify your code accordingly, until you've passed all unit tests.

The following are some examples of the assersion error. 

- AssertionError: Matrix A shouldn't be modified
  + Your augmentMatrix modifies matrix A. 
  
- AssertionError: Matrix A is singular
  + Your gj_Solve doesn't return None when A is singular. 
- AssertionError: Matrix A is not singular
  + Your gj_Solve returns None when A is not singular.
- AssertionError: x have to be a two-dimensional Python list
  + Your gj_Solve returns with incorrect data structure. X should be a two a list of lists. 

- AssertionError: Regression result isn't good enough
  + Your gj_Solve has too much error. 

# Project submission
Before submission, please ensure that your project meets all the requirements in this [rubric](https://review.udacity.com/#!/rubrics/854/view). Reviewer will give reviews according to this rubric.

You should submit the follow four files. Please pay attention to the file names and file types. 

1. `linear_regression_project.ipynb`: the ipynb file with your code and answers. 

3. `linear_regression_project.html`: the html file exported by Jupyter notebook.

3. `linear_regression_project.py`: the python file exported by Jupyter notebook.

2. `proof.pdf`: the pdf file with your proof. （If you use LATEX in ipython notebook to write the proof, you don't need to submit this file. ）

5. Please DO NOT submit any other files.

You can use Github or upload zip file to submit the project. If you use Github, please include all your files in the repo. If you submit with zip file, please compress all your files in `submit.zip` and then upload. 
