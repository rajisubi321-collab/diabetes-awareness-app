def chatbot_response(message: str) -> str:
    q = (message or "").lower().strip()

    if any(word in q for word in ["emergency", "unconscious", "can't breathe", "chest pain"]):
        return ("If this may be an emergency, contact local emergency services immediately. "
                "This chatbot cannot assess emergencies.")
    if "what is diabetes" in q or q == "diabetes":
        return ("Diabetes is a condition in which blood glucose is too high because the body "
                "does not make enough insulin or cannot use insulin effectively. A healthcare "
                "professional can advise about testing and diagnosis.")
    if "bmi" in q:
        return ("BMI is calculated from weight in kilograms divided by height in metres squared. "
                "It is a screening measure and does not diagnose diabetes or fully describe health.")
    if any(w in q for w in ["prevent", "reduce risk", "lower risk", "avoid diabetes"]):
        return ("Helpful habits can include regular physical activity suited to your abilities, "
                "balanced meals with vegetables and fibre-rich foods, adequate sleep, and attending "
                "recommended health checks. Personal advice should come from a qualified clinician.")
    if any(w in q for w in ["exercise", "physical activity", "walking"]):
        return ("Regular movement can support metabolic health. Choose an activity appropriate for "
                "your health and ability; ask a clinician before making major changes if you have concerns.")
    if any(w in q for w in ["food", "diet", "eat", "fruit", "vegetable"]):
        return ("Aim for varied, balanced meals that fit your culture, budget, and health needs. "
                "A registered dietitian or clinician can provide personalised advice.")
    if any(w in q for w in ["symptom", "thirst", "urination", "tired"]):
        return ("Possible diabetes symptoms can include increased thirst, frequent urination, fatigue, "
                "and blurred vision, but some people have no obvious symptoms. Arrange medical evaluation "
                "if you are concerned; only appropriate testing can diagnose diabetes.")
    if any(w in q for w in ["score", "prediction", "result", "risk"]):
        return ("Your app's score reflects the model's output for patterns in its source dataset. "
                "It is not a diagnosis or a validated probability of developing diabetes. Discuss screening "
                "and personal risk with a qualified healthcare professional.")
    if any(w in q for w in ["family history", "genetic"]):
        return ("Family history can be relevant to diabetes risk. Tell your healthcare professional about "
                "it when discussing screening. This prototype does not include family history in its trained model.")
    if any(w in q for w in ["doctor", "screen", "test", "diagnos"]):
        return ("A healthcare professional can recommend appropriate screening based on your circumstances. "
                "Do not use this app to diagnose yourself or change treatment.")
    return ("I can answer general questions about diabetes, BMI, lifestyle habits, symptoms, screening, "
            "and how this app's model output should be interpreted. I cannot diagnose or provide personalised treatment.")
