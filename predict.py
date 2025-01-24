import json

def make_prediction(data, question):
    return (data["theta0"] * question) + data["theta1"]

def main():
    try:
        with open('parameters.json', 'r') as file:
            data = json.load(file)
        question = input("Quelle kilometrage vous interesse ? ")
        question = float(question)
        prediction = make_prediction(data, question)
        print(f"Le prix pour une voiture qui a {question}km est {prediction}$")
    except KeyboardInterrupt:
        return
    except Exception as error:
        print(f"Error: {error}")

main()