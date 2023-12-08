from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__, static_folder="static")

# The URL of FastAPI API
api_url = "http://127.0.0.1:8000/models/"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the text entered by the user
        text = request.form.get("text")

        # Build the URL with the text as a parameter
        url = f"{api_url}?payload={text.replace(' ', '%20')}"

        # Call your FastAPI API
        response = requests.post(url)

        if response.status_code == 200:
            # Get the results from the API
            data = response.json()
            prediction = data["data"]["prediction"]
            model_type = data["data"]["model-type"]

            return render_template("index.html", prediction=prediction, model_type=model_type, text=text)
        else:
            error_message = "Error when calling API."
            return render_template("index.html", error_message=error_message, text=text)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)
