This is the repo for the machine learning website.

In order to access the website, you will need to have streamlit installed. The code utilizes the Streamlit API in order to create and host a website locally. 

Best practice is to host this code into a virtual environment. Use any preferred way.
  The recommended method is:
  
    1) Open command line and go to your preferred directory
    
    2)  Now in the directory type in:
          python -m venv <environment name> (Windows)
                        or
          python -m virtualenv <environment name> (Mac & Linux)
    
    
    3) Open the directory that has your virtual environment in any IDE of your choice (VSCode Recommended)

    4) Open a terminal in your IDE, and it should automatically activate your virtual environemnt. At this point make sure that this repo is in the same directory 
    as the virtual directory

    5) Finally, in the terminal of VSCode where the virtual environment is hosted, run the command

        streamlit run app.py


      This should automatically host and run a website that displays the web app
      
      
