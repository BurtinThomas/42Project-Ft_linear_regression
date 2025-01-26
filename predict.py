import json

def make_prediction(data, question):
    return (data["theta0"] * question) + data["theta1"]

def main():
    try:
        with open('parameters.json', 'r') as file:
            data = json.load(file)
        if (data["theta0"] == 0 or data["theta1"] == 0):
            raise ValueError("First, you have to train the model")
        question = float(input("Quelle kilometrage vous interesse ? "))
        if question < 0:
            raise ValueError("The model accept only positiv value")
        prediction = make_prediction(data, question)
        print(f"Le prix pour une voiture qui a {question}km est {prediction}$")
    except KeyboardInterrupt:
        return
    except Exception as error:
        print(f"Error: {error}")

main()