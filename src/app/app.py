from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__, static_folder="static")

# L'URL de votre API FastAPI
api_url = "http://127.0.0.1:8000/models/"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Récupérer le texte saisi par l'utilisateur
        text = request.form.get("text")

        # Construire l'URL avec le texte comme paramètre
        url = f"{api_url}?payload={text.replace(' ', '%20')}"

        # Appeler votre API FastAPI
        response = requests.post(url)

        if response.status_code == 200:
            # Obtenir les résultats de l'API
            data = response.json()
            prediction = data["data"]["prediction"]
            model_type = data["data"]["model-type"]

            return render_template("index.html", prediction=prediction, model_type=model_type, text=text)
        else:
            error_message = "Erreur lors de l'appel à l'API."
            return render_template("index.html", error_message=error_message, text=text)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
