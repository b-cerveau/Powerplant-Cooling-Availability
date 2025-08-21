# Powerplant-Cooling-Availability
This repository contains the supporting code to the simulation of climate change on impact on thermoelectrical generation.

## Dependencies & Documentation Access
------------------------------------

To view the documentation locally, you’ll first need to set up a Python environment and install the required modules.

1) Create and activate a virtual environment

   Using conda (recommended):
   
       conda create -n your_env_name python=3.11
   
       conda activate your_env_name

   Using venv (built into Python):
   
       python -m venv .venv
   
       # Mac/Linux
       source .venv/bin/activate
   
       # Windows
       .venv\Scripts\activate

3) Navigate to your designated directory e.g. Powerplant-Cooling-Availability
   
       cd /path/to/Powerplant-Cooling-Availability

4) Install dependencies

   Separate installs:
   
       pip install -r docdependencies.txt
       pip install -r codedependencies.txt

   Install both at once:
   
       pip install -r docdependencies.txt -r codedependencies.txt

5) Build and serve the documentation
   
       mkdocs serve

   This will start a local web server.  
   Open your browser and go to:
       http://127.0.0.1:8000

6) Stop the local server

   Press Ctrl + C in the terminal running `mkdocs serve`.

   If that doesn’t work:

   Mac/Linux:
       pkill -f mkdocs

   Windows (PowerShell):
       Stop-Process -Name python -Force
   (Only do this if you’re sure the process is the MkDocs server.)
