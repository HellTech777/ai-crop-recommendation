🛠️ Run the App Locally
To run the crop recommendation app on your local machine, first make sure you have Python 3.8 or higher installed. 
Clone the repository to your computer using Git, then navigate into the project folder. 
(Optionally, you can create a virtual environment to isolate dependencies.) 
Install all required packages using 
pip install -r requirements.txt. 
Once setup is complete, start the FastAPI server by running 
uvicorn agri_app.main:app --reload 
in your terminal. 
This will launch the app locally at 
http://127.0.0.1:8000, 
where you can access the crop recommendation form, select a city, and view results in your browser.
