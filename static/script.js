const API_URL = "http://127.0.0.1:5000/predict";

document.getElementById("analyzeBtn").addEventListener("click", analyzeSentiment);

async function analyzeSentiment() {
  const text = document.getElementById("inputText").value;
  if (!text) {
    alert("Please enter some text.");
    return;
  }

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    document.getElementById("result").innerHTML =
      `Prediction: ${data.prediction} <br> Probability: ${data.probability.toFixed(2)}`;

  } catch (error) {
    alert("Failed to connect to backend: " + error.message);
  }
}
