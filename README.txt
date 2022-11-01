*************************************************************************************************
*												*
*	Brian Nason - Student Number 001003011							*
*	contact: bnason1@wgu.edu								*
*												*
*	Capstone project for B.S. in Computer Science						*
*												*
*	Topic: Creating a Random Forest Regression model to estimate the cost/price of a house 	*
*	in King	County, Washington based on user inputs of the specs of a house.		*
*												*
*--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--*
*												*
*	Language: Python									*
*	License: Open-source									*
*	Version: 1.2.1										*
*												*
*--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--*
*												*
*	Package File list:									*
*	1) capstoneProject.py									*
*	2) kc_house_data.csv									*
*	3) README.txt										*
*												*
*--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--/--*
*												*
*	Project Dependencies/libraries:								*
*	1) Python (3.9.13 or better)								*
*		https://www.python.org/downloads/						*
*	2) numpy (1.23.2 or better)								*
*		https://numpy.org/install/							*
*	3) pandas (1.4.0 or better)								*
*		https://pandas.pydata.org/docs/getting_started/install.html			*
*	4) matplotlib (3.6.1 or better)								*
*		https://matplotlib.org/stable/users/installing/index.html			*
*	5) scikit-learn	(1.1.2 or better)							*
*		https://scikit-learn.org/stable/install.html					*
*	6) pydot (1.4.2 or better)								*
*		https://pypi.org/project/pydot/							*
*	7) PySimpleGUI (4.6.04 or better)							*
*		https://www.pysimplegui.org/en/latest/						*
*												*
*************************************************************************************************

This README file is broken down into 8 sections:

I.	GitHub Link to Project:
II.	Unzipping instructions for project files:
III.	Instructions to run project through CLI/Terminal:
IV.	Troubleshooting for CLI/Terminal running:
V.	Instructions to run project through Visual Studios 2022:
VI.	Troubleshooting for project execution:
VII.	Basic GUI instructions:
VIII.	FAQ’s:

*************************************************************************************************

I.	GitHub Link to Project:

This project is hosted on GitHub at the following link:

https://github.com/BrianENason/projectBlueLight

Click on the green "Code" button and select "Download Zip" from the dropdown.

Please note: This repository is shared publicly for your ease of access. Because of this, the 
name of the repo is "projectBlueLight" to prevent any plagiarism from other students doing a 
keyword search for “capstone”, “WGU”, “C964”, etc.

*************************************************************************************************

II.	Unzipping instructions for project files:

Prerequisite: You will need to have a file archiver/compression program on your system to unzip 
the compressed project files like “WinZip”. 

If you need a solidly-built, open-sourced option, consider using 7-zip. Link to download and 
install instructions here:

https://www.7-zip.org/

Once you have your .zip program installed, simply right-click on the .zip file and extract the 
files to a folder that is on your main drive and easy to get to (desktop, documents, and downloads 
are all good options)

*************************************************************************************************

III.	Instructions to run project through CLI/Terminal:

Open your Command Line Interface (Windows) or your terminal (Mac).

Check to see that you have Python already installed on your computer by typing (without quotes):

	“python --version" (for Windows) or 
	“python -V” (for Mac) 

If this returns anything like version number or location, then you are good to go. If it comes 
back with something like “file not found”, then you will need to install Python on your system 
first using this link to download and follow a step-by-step install for your system:

	https://www.python.org/downloads/

Once Python is installed and configured on your computer, open a fresh CLI or terminal.

You now need to verify that "pip" was installed with your Python package. To do this, type in
you CLI or Terminal (without quotes):

	"pip --version" (for Windows) or
	"pip -V" (for Mac)

If it was installed, you will get a "Version Number" returned. If not, you will need to install
it manually following the instructions at this link:

	https://pip.pypa.io/en/stable/installation/

Once you have verified that you have Pip installed, you can discover what libraries/dependencies
you already have by typing the following command (no quotes):

	"pip list"

This will output all of the existing libraries and dependencies you have associated with your
Python. You can check this list against the "Project Dependencies/libraries" section at the top
of this README file.

For the libraries you don't have yet, you are going to have to install them using "pip". REALIZE: 
the order matters as some libraries have a dependency on other libraries already installed before 
they can be, so execute them in this order below (omit the quotes):

	"pip install numpy”
	“pip install pandas”
	“pip install matplotlib”
	“pip install scikit-learn”
	“pip install pydot”
	“pip install PySimpleGUI”

NOTE: if you have trouble installing any of the packages, copy and paste the links from the 
"Project Dependencies/libraries" section at the top of this README into your browser and follow 
their instructions.

Verify that all the libraries have been installed by using the "pip list" command to see all the
associated libraries.

Once the libraries have all been installed, navigate your CLI or Terminal to the folder with the 
project files.

Once inside the folder, you are going to type the following command (no quotes) and hit enter:

	“python capstoneProject.py”

This will kick off the program and you will be good to go.

*************************************************************************************************

IV.	Troubleshooting for CLI/Terminal running:

In the unlikely event that you have issues running the code from the CLI or Terminal, try the 
following:

1) Close out your CLI/Terminal and start a fresh session
2) Try re-running the "python capstoneProject.py" command especially if the error happens on your
first ever attempt at running the code).
3) Follow the Error instructions to rectify the problem
4) Ensure the project folder is on the same drive as the python install
5) Check you path variable (Windows) by using the following step-by-step instructions at this link:

	https://www.educative.io/answers/how-to-add-python-to-path-variable-in-windows

6) If all else fails, consider using the Visual Studio 2022 method in the next section.

*************************************************************************************************

V.	Instructions to run project through Visual Studio 2022:

An alternative to the CLI/terminal running of the project is to install “Visual Studio 2022” and run
the project directly from the IDE. You can get the download from the following link:
	
	https://visualstudio.microsoft.com/

Once Visual Studio is installed and configured, you will:

Right-click on the unzipped project folder and navigate to the option “Open with Visual Studio”. 

When the project finishes loading into the IDE, double-click on the “capstoneProject.py” file to 
bring the code to the main screen.
NOTE: If you don’t see the solution explorer on the screen with all the folder’s files in it, use the
following shortcut to bring it up: 

Ctrl+Alt+L

Once the capstoneFinal.py is opened in the IDE Editor Window, you are going to click the small gift 
icon in the upper toolbar just to the right of the Python Environment dropdown to bring up the Visual
Studio Package Manager.

Inside the package manager (likely showing on the screen where the "Solution Explorer" just was), you
are going to check if the necessary libraries are installed. Using the manager's search bar, type in 
the following libraries one-by-one:

	1) numpy
	2) pandas
	3) matplotlib
	4) scikit-learn
	5) pydot
	6) PySimpleGui

NOTE: If you already have them in your package manager, then they will appear as an option. If not, the 
top option/result from the search will be “Run Command: pip install xxx”. Click this option.

With all the libraries linked, you should be able to now click on the green arrow in the top toolbar 
to run the program.

*************************************************************************************************

VI.	Troubleshooting for Visual Studios 2022 running:

In the unlikely event that you have issues running the code from the Visual Studio 2022 IDE, try
the following solutions:

1) Verify that the project folder contains a .vs folder. This is used by Visual Studio for its 
inner-workings and processing.
2) Make sure "Current Document(capstoneProject.py)" is selected in the "Startup Item" dropdown 
attached to the solid green play button.
3) If the "capstoneProject.py" can't be found in the dropdown, follow these steps:
	1) Click: File > New > Project From existing code
	2) Select type of project and click Next. 
	3) Enter "capstoneProject.py" and click Finish.

*************************************************************************************************

VII.	Basic GUI instructions:

The GUI (Graphical User Interface) for the project consists of 4 tabs. Each tab has a specific role
in the program.

1) Main Page - This is where the user enters the specs about the house they are pricing out. 
	a. All fields have been initialized with some value and all fields are necessary before the 
		estimation can be created by the model.
	b. Spaces, special characters, and letters are not allowed in any of the input boxes except
		for the "Name This Collection" field. If a user accidently enters illegal characters, 
		the program will display an error popup and will not process the information until 
		the issue is fixed.
	c. Every time a successful estimation is returned, the program will log those specs and their
		result using the "Name This Collection" field. Therefore, a new name will have to be
		entered by the user each time they want to run some specs.
2) Graphs - This will display all the graphs and a brief description about them on demand. Simply
select a graph and it will be displayed. 
	a. There is a special button "Clear Charts" that can be clicked in case the user wishes to
		remove all graphs from the visual display. This is NOT destructive. The charts can
		still be recalled by selecting them.
3) About - This is hard-coded information about the program and the program's creator (me).
4) Log - This serves 3 purposes:
	a. It records all interactions between the user and the GUI (very useful for debugging).
	b. It saves and displays each house's stats and the resulting estimation.
	c. It shows the program's ability to capture and record each interaction. This data could be
		saved and used in production to increase the accuracy of the model - especially if
		the user's actual/final cost/price are recorded.

*************************************************************************************************

VIII.	FAQ’s:

Q. How long did this take to develop?
A. Finding a dataset that I could create a useful project out of took the most time (almost 3 weeks).
	Once I found this dataset, I was able to code it out in about a week, complete testing in just
	a couple days, and complete the documentation in about 5 days. So overall, it took 6 weeks from
	opening the course to final submission.

Q. How many data sets/ alternative projects did you work on before arriving at this solution?
A. I went through 5 different data sets and brought them all to various levels of completion. They 
	were ultimately rejected for various reasons.

Q. Why is the program not in an executable form?
A. The "lore" behind this is that the code will be integrated into a fictitious website. Therefore,
	creating it as a stand-alone code would not be a realistic solution.

Q. Is that really the real reason?
A. Partially. Another reason it is not in an executable file is because the pandas library I used
	was not converting properly, regardless of what I did. I wasted a lot of time on trying to 
	get it to work, but in the end, I decided the effort was not worth the cost.

Q. Why did you not release it in an online runtime environment like a binder or a notebook?
A. I really like using a GUI for user interaction, and none of the online options I could find would
	allow me to use my code the way I intend it.

Q. But, some do support GUI interactions. Did you know that?
A. Yes, but those that do were either pay-to-play or too complicated to learn for this project.

Q. Why did you choose Python?
A. I actually started this project fully intending to create it in Java, but I quickly realized that 
	there are a lot more useful libraries for data science that are written in Python. If this
	project had not required the use of Machine Learning, I would have written it in Java.

Q. How many hours do you think you spent on the development of this?
A. More than I would care to admit. But if I had to low-ball it, I would say 100 hours give-or-take.
